import streamlit as st
import os
import re
from pathlib import Path
import tempfile # To save uploaded files temporarily
import shutil # To remove temp directory
import natsort # For natural sorting
import time # For progress simulation
from PIL import Image # For artwork check
from pydub import AudioSegment # For audio spec check
from pydub.utils import mediainfo # For audio spec check
import numpy as np # For audio sample manipulation
from collections import defaultdict # For grouping samples
from spellchecker import SpellChecker # For spelling check
import pyperclip # For copy to clipboard
import soundfile as sf # For LUFS calculation audio loading
import pyloudnorm as pyln # For LUFS calculation
import pandas as pd # For displaying the table

# --- Constants ---
VALID_SUBFOLDERS = {"Loops", "One Shots"}
DEMO_SUFFIX = "_Demo.wav"
ARTWORK_SUFFIX = "_Artwork"
VALID_ARTWORK_EXT = {".jpg", ".jpeg"} # Allow .jpeg as well, common alias for jpg
TARGET_ARTWORK_EXT = ".jpg" # Explicitly target jpg
MAX_ARTWORK_SIZE_MB = 4
MIN_ARTWORK_DIM = 1000
# Only allow WAV files for processing and spec checks
VALID_AUDIO_EXT = {".wav"}
TARGET_SAMPLE_RATE = 44100
TARGET_BIT_DEPTH = 24
TARGET_CHANNELS = 2
SILENCE_THRESHOLD_DBFS = -60.0 # Threshold for silence detection (in dBFS)
ZERO_CROSSING_THRESHOLD = 0.001 # Amplitude threshold for zero crossing
BAR_LENGTH_TOLERANCE = 0.001 # Tolerance for whole bar length check (as percentage of a bar)
BPM_REGEX = re.compile(r"_(\d+)bpm")
# Regex for sequence number: _<digits>_ OR _<digits>.<extension>
SEQUENCE_REGEX = re.compile(r"_(\d+)_|_(\d+)\.(?=[^.]*$)")
# Regex for key: _key<key_notation>
KEY_REGEX = re.compile(r"_key([A-Ga-g][#b♭]?[mM]?)") # More robust key regex
# Musical key order (using numeric values for sorting)
# Minor keys get a fractional part for sorting after major
KEY_ORDER = {
    "C": 0, "Cm": 0.1,
    "C#": 1, "Db": 1, "C#m": 1.1, "Dbm": 1.1,
    "D": 2, "Dm": 2.1,
    "D#": 3, "Eb": 3, "D#m": 3.1, "Ebm": 3.1,
    "E": 4, "Em": 4.1,
    "F": 5, "Fm": 5.1,
    "F#": 6, "Gb": 6, "F#m": 6.1, "Gbm": 6.1,
    "G": 7, "Gm": 7.1,
    "G#": 8, "Ab": 8, "G#m": 8.1, "Abm": 8.1,
    "A": 9, "Am": 9.1,
    "A#": 10, "Bb": 10, "A#m": 10.1, "Bbm": 10.1,
    "B": 11, "Bm": 11.1
}
# Spelling Constants
SPELLCHECK_IGNORE_WORDS = {"Metastarter", "Cymatics", "Ableton", "Synth", "Serum", "Vital", "Phaseplant"} # Add common technical/brand terms
# Regex to find parts to REMOVE before splitting into words for spellcheck
SPELLCHECK_REMOVE_PATTERNS = re.compile(r"_key[A-Ga-g][#b♭]?[mM]?|_\d+bpm|_\d+_|_\d+$|_+Demo$|_Artwork$", re.IGNORECASE)
# Regex to split remaining text into words (CamelCase, underscore/space separated)
SPELLCHECK_SPLIT_WORDS_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|\d+")

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Make Scott Proud!")

# --- State Management ---
if 'dropped_items' not in st.session_state:
    st.session_state.dropped_items = None # Store UploadedFile objects
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_display_data' not in st.session_state:
    # Structure: {subfolder_path: Path, files: [{path: Path, display_name: str, issues: list, ...}]}
    st.session_state.file_display_data = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'reset_app' not in st.session_state:
    st.session_state.reset_app = False
if 'loudness_results' not in st.session_state:
    st.session_state.loudness_results = [] # Store dicts: {filename: str, lufs: float | str}

# --- Helper Functions ---

def natural_sort_key(path):
    """Provides a key for natural sorting of Path objects."""
    return natsort.natsort_keygen()(path.name)

def get_files_from_item(item_path: Path) -> list[Path]:
    """Recursively get all files from a path (file or directory)."""
    all_files = []
    if item_path.is_file():
        # If a single file is dropped, its 'parent' for grouping will be its actual parent
        all_files.append(item_path)
    elif item_path.is_dir():
        # If a directory is dropped, we treat IT as the 'subfolder' key later
        # But we still need all files inside it recursively
        for root, _, files in os.walk(item_path):
            for file in files:
                file_path = Path(root) / file
                # Simple check to ignore hidden files (like .DS_Store)
                if not file.startswith('.'):
                    all_files.append(file_path)
    return all_files

def add_issue(file_data, issue_type):
    """Adds an issue type to a file's data if not already present."""
    if issue_type not in file_data['issues']:
        file_data['issues'].append(issue_type)

def get_normalized_samples(audio: AudioSegment) -> np.ndarray | None:
    """Gets audio samples as a numpy array normalized to [-1.0, 1.0]."""
    try:
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        # Normalize based on sample width (bytes)
        max_val = (2**(audio.sample_width * 8 - 1))
        samples /= max_val
        return samples
    except Exception as e:
        print(f"Error getting normalized samples: {e}")
        return None

def get_sorted_messages(messages: list[str]) -> list[str]:
    """Sorts messages by type (✅, ❔, ❌)."""
    success_msgs = sorted([m for m in messages if m.startswith("✅")])
    warning_msgs = sorted([m for m in messages if m.startswith("❔")])
    error_msgs = sorted([m for m in messages if m.startswith("❌")])
    return success_msgs + warning_msgs + error_msgs

# --- QC Check Functions ---

def validate_folder_names(subfolder_paths: list[Path]) -> list[str]:
    """Checks if subfolder names are valid."""
    messages = []
    invalid_subfolders = []
    # Only check folders that were directly dropped or are direct children of dropped folders
    # This logic might need refinement based on how users drop things.
    # For now, we check the keys of file_display_data, assuming they represent the intended subfolders.
    for sf_path in subfolder_paths:
        if sf_path.is_dir() and sf_path.name not in VALID_SUBFOLDERS:
            invalid_subfolders.append(sf_path.name)

    if not invalid_subfolders:
        messages.append("✅ Folder names")
    else:
        messages.append(f"❌ Invalid subfolder names found (must be {', '.join(VALID_SUBFOLDERS)})")
        for name in invalid_subfolders:
            messages.append(f"    - Invalid: '{name}'")
    return messages

def validate_filename_spaces(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks for consecutive spaces or spaces around underscores in filenames."""
    messages = []
    updated_file_data = file_display_data
    has_any_space_issue = False

    space_around_underscore_re = re.compile(r"\s_|_\s")
    consecutive_spaces_re = re.compile(r"\s{2,}")

    for subfolder, files in updated_file_data.items():
        for file_data in files:
            filename = file_data['display_name']
            file_issues = []
            if consecutive_spaces_re.search(filename):
                file_issues.append("consecutive spaces")
                add_issue(file_data, 'spacing')
                has_any_space_issue = True
            if space_around_underscore_re.search(filename):
                file_issues.append("space around underscore")
                add_issue(file_data, 'spacing')
                has_any_space_issue = True

            if file_issues:
                messages.append(f"❌ Spacing: '{filename}' has {', '.join(file_issues)}")

    if not has_any_space_issue:
        messages.append("✅ No filename spacing issues")

    return updated_file_data, messages

def validate_demo_file(file_display_data: dict) -> tuple[dict, list[str]]:
    """Validates that at least one correctly named demo file exists."""
    messages = []
    updated_file_data = file_display_data
    actual_demos = []

    for sf_key, files in updated_file_data.items():
        for file_data in files:
            filename = file_data['display_name']
            # Check for actual, correctly named demos
            if filename.endswith(DEMO_SUFFIX):
                actual_demos.append((file_data, sf_key))

    if len(actual_demos) >= 1:
        # Found at least one valid demo
        first_demo_name = actual_demos[0][0]['display_name']
        messages.append(f"✅ Demo file found: '{first_demo_name}" + ("' (and possibly others)" if len(actual_demos) > 1 else "'"))
        # Mark the first found demo as ok (or all if needed, depends on UI goal)
        add_issue(actual_demos[0][0], 'demo_ok') # Add a positive marker if needed
    else:
        # No file ends with _Demo.wav, check if any contained " - " as a hint
        potential_demos = []
        for sf_key, files in updated_file_data.items():
            for file_data in files:
                 if " - " in file_data['display_name']:
                      potential_demos.append(file_data)

        if potential_demos:
             messages.append(f"❌ Missing Demo file (None ending in '{DEMO_SUFFIX}', but found files with ' - ')")
             for file_data in potential_demos:
                  add_issue(file_data, 'demo_missing_suffix')
        else:
             messages.append(f"❌ Missing Demo file (None ending in '{DEMO_SUFFIX}')")

    return updated_file_data, messages

def validate_artwork(file_display_data: dict) -> tuple[dict, list[str]]:
    """Validates that at least one valid artwork file exists (.jpg/.jpeg only)."""
    messages = []
    updated_file_data = file_display_data
    found_valid_artwork = False
    first_valid_artwork_name = ""

    for subfolder, files in updated_file_data.items():
        if found_valid_artwork: break # Optimization: stop checking once one is found
        for file_data in files:
            f_path = file_data['path']
            filename = file_data['display_name']
            issues = []

            # Check if it's a JPG/JPEG file first
            if f_path.suffix.lower() not in VALID_ARTWORK_EXT:
                continue # Skip non-jpg/jpeg files

            # Now check naming convention for this JPG/JPEG
            expected_suffix_jpg = ARTWORK_SUFFIX + ".jpg"
            expected_suffix_jpeg = ARTWORK_SUFFIX + ".jpeg"
            is_correctly_named = (filename.lower().endswith(expected_suffix_jpg.lower()) or
                                  filename.lower().endswith(expected_suffix_jpeg.lower()))

            if not is_correctly_named:
                add_issue(file_data, 'artwork_naming')
                # Don't report as overall error yet, just mark this file
                continue # Skip spec checks if name is wrong

            # --- Check Specs (only if name is correct) ---
            try:
                # Check file size
                size_bytes = f_path.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                if size_mb > MAX_ARTWORK_SIZE_MB:
                    issues.append(f"size exceeds {MAX_ARTWORK_SIZE_MB}MB")
                    add_issue(file_data, 'artwork_size')

                # Check dimensions and squareness
                with Image.open(f_path) as img:
                    if img.format not in ["JPEG"]:
                         issues.append(f"format is '{img.format}', not JPEG")
                         add_issue(file_data, 'artwork_format')

                    width, height = img.size
                    if width < MIN_ARTWORK_DIM or height < MIN_ARTWORK_DIM:
                        issues.append(f"dims < {MIN_ARTWORK_DIM}x{MIN_ARTWORK_DIM}")
                        add_issue(file_data, 'artwork_dims')
                    if width != height:
                        issues.append("not square")
                        add_issue(file_data, 'artwork_square')

            except FileNotFoundError:
                issues.append("file not found")
                add_issue(file_data, 'artwork_error')
            except Exception as e:
                issues.append(f"read error: {e}")
                add_issue(file_data, 'artwork_error')

            # --- Decision --- #
            if not issues:
                # This file is correctly named AND meets all specs
                found_valid_artwork = True
                first_valid_artwork_name = filename
                add_issue(file_data, 'artwork_ok') # Add positive marker if needed
                break # Stop checking this subfolder's files
            else:
                 # File was correctly named but failed spec checks
                 # Issue already added, maybe add a specific message for this file?
                 messages.append(f"❌ Artwork Specs: '{filename}' failed checks: {', '.join(issues)}")

    # --- Final Message --- #
    if found_valid_artwork:
        messages.append(f"✅ Artwork found: '{first_valid_artwork_name}'")
    else:
        # No valid artwork found after checking all files
        messages.append(f"❌ Missing valid Artwork file ({ARTWORK_SUFFIX}{TARGET_ARTWORK_EXT})")
        # Optionally, add warnings about files that were close (e.g., named correctly but failed specs)

    return updated_file_data, messages

def validate_audio_specs(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks audio file specs (.wav only: format, sample rate, bit depth, channels)."""
    messages = []
    updated_file_data = file_display_data
    has_any_spec_issue = False
    checked_files = 0
    found_non_wav = False

    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            filename = file_data['display_name']
            original_suffix = Path(filename).suffix.lower() # Check suffix from original name

            # Report non-WAV audio files first
            if original_suffix in {".aif", ".aiff", ".mp3", ".aac", ".m4a"}:
                messages.append(f"❌ Audio Format: '{filename}' is not a WAV file (found {original_suffix})")
                add_issue(file_data, 'audio_format')
                has_any_spec_issue = True
                found_non_wav = True
                continue

            # Only process files with the target .wav extension further
            if original_suffix not in VALID_AUDIO_EXT:
                continue

            checked_files += 1
            issues = []
            try:
                # Use soundfile.info() to get accurate metadata
                info = sf.info(str(f_path))

                # Check sample rate
                if info.samplerate != TARGET_SAMPLE_RATE:
                    issues.append(f"not {TARGET_SAMPLE_RATE} Hz ({info.samplerate} Hz)")
                    add_issue(file_data, 'audio_samplerate')
                    has_any_spec_issue = True

                # Check bit depth via subtype
                # Common subtypes: PCM_16, PCM_24, PCM_32, FLOAT, DOUBLE
                is_24_bit = 'PCM_24' in info.subtype
                if not is_24_bit:
                    issues.append(f"not {TARGET_BIT_DEPTH}-bit (subtype: {info.subtype})")
                    add_issue(file_data, 'audio_bitdepth')
                    has_any_spec_issue = True

                # Check channels
                if info.channels != TARGET_CHANNELS:
                    issues.append(f"not {TARGET_CHANNELS}-channel stereo ({info.channels} channels)")
                    add_issue(file_data, 'audio_channels')
                    has_any_spec_issue = True

            except FileNotFoundError:
                issues.append("file not found")
                add_issue(file_data, 'audio_error')
                has_any_spec_issue = True
            except Exception as e:
                if "ffmpeg" in str(e).lower() or "ffprobe" in str(e).lower():
                     issues.append(f"error reading audio (possible missing FFmpeg/libsndfile or corrupted file): {e}")
                else:
                    issues.append(f"error reading audio: {e}")
                add_issue(file_data, 'audio_error')
                has_any_spec_issue = True

            if issues:
                messages.append(f"❌ Audio Specs '{filename}': {', '.join(issues)}")

    # Report overall status
    if not has_any_spec_issue and checked_files > 0:
        messages.append("✅ Audio file specs correct (for .wav)")
    elif checked_files == 0 and not found_non_wav:
        messages.append("❔ No .wav files found to check specs.")

    return updated_file_data, messages

def validate_clipping(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks audio files for clipping."""
    messages = []
    updated_file_data = file_display_data
    has_clipping = False
    checked_files = 0
    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            if f_path.suffix.lower() not in VALID_AUDIO_EXT:
                continue
            checked_files += 1
            try:
                audio = AudioSegment.from_file(str(f_path))
                samples = get_normalized_samples(audio)
                if samples is not None and np.max(np.abs(samples)) >= 1.0:
                    messages.append(f"❌ Clipping detected in '{file_data['display_name']}'")
                    add_issue(file_data, 'audio_clipping')
                    has_clipping = True
            except Exception as e:
                messages.append(f"⚠️ Error checking clipping for '{file_data['display_name']}': {e}")
                add_issue(file_data, 'audio_error_clipping')
    if not has_clipping and checked_files > 0:
        messages.append("✅ No clipping detected")
    return updated_file_data, messages

def validate_silence(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks if audio files are effectively silent."""
    messages = []
    updated_file_data = file_display_data
    has_silence = False
    checked_files = 0
    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            if f_path.suffix.lower() not in VALID_AUDIO_EXT:
                continue
            # Avoid checking demo files for silence
            if DEMO_SUFFIX.lower() in file_data['display_name'].lower():
                continue
            checked_files += 1
            try:
                audio = AudioSegment.from_file(str(f_path))
                # Check dBFS. Extremely low values indicate silence.
                if audio.dBFS < SILENCE_THRESHOLD_DBFS:
                    messages.append(f"❌ Silent track detected: '{file_data['display_name']}' ({audio.dBFS:.2f} dBFS)")
                    add_issue(file_data, 'audio_silence')
                    has_silence = True
            except Exception as e:
                messages.append(f"⚠️ Error checking silence for '{file_data['display_name']}': {e}")
                add_issue(file_data, 'audio_error_silence')
    if not has_silence and checked_files > 0:
        messages.append("✅ No silent tracks detected")
    return updated_file_data, messages

def validate_zero_crossings(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks if audio files start and end near zero crossing."""
    messages = []
    updated_file_data = file_display_data
    has_crossing_issue = False
    checked_files = 0
    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            if f_path.suffix.lower() not in VALID_AUDIO_EXT:
                continue
            # Avoid checking demo files
            if DEMO_SUFFIX.lower() in file_data['display_name'].lower():
                continue
            checked_files += 1
            try:
                audio = AudioSegment.from_file(str(f_path))
                samples = get_normalized_samples(audio)
                if samples is None or len(samples) < 2:
                    raise ValueError("Could not get sufficient samples")

                start_ok = True
                end_ok = True
                channels = audio.channels

                # Check start samples (first sample of each channel)
                for i in range(channels):
                    if abs(samples[i]) > ZERO_CROSSING_THRESHOLD:
                        start_ok = False
                        break

                # Check end samples (last sample of each channel)
                for i in range(channels):
                     if abs(samples[-channels + i]) > ZERO_CROSSING_THRESHOLD:
                        end_ok = False
                        break

                if not start_ok or not end_ok:
                    issue_desc = []
                    if not start_ok: issue_desc.append("start")
                    if not end_ok: issue_desc.append("end")
                    messages.append(f"❌ Zero Crossing: '{file_data['display_name']}' does not {' and '.join(issue_desc)} on zero crossing.")
                    add_issue(file_data, 'audio_zero_crossing')
                    has_crossing_issue = True

            except Exception as e:
                messages.append(f"⚠️ Error checking zero crossing for '{file_data['display_name']}': {e}")
                add_issue(file_data, 'audio_error_crossing')
    if not has_crossing_issue and checked_files > 0:
        messages.append("✅ Zero crossings OK")
    return updated_file_data, messages

def validate_sample_lengths(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks if audio file duration matches whole bars based on BPM in filename."""
    messages = []
    updated_file_data = file_display_data
    has_length_issue = False
    checked_files = 0

    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            filename = file_data['display_name']

            if f_path.suffix.lower() not in VALID_AUDIO_EXT:
                continue
            if DEMO_SUFFIX.lower() in filename.lower():
                continue

            bpm_match = BPM_REGEX.search(filename)
            if not bpm_match:
                continue # Skip files without BPM for this check

            checked_files += 1
            bpm = int(bpm_match.group(1))

            try:
                audio = AudioSegment.from_file(str(f_path))
                duration_sec = audio.duration_seconds

                if bpm <= 0: raise ValueError("Invalid BPM")

                seconds_per_bar = (60.0 / bpm) * 4.0
                if seconds_per_bar <= 0: raise ValueError("Invalid seconds per bar")

                num_bars = duration_sec / seconds_per_bar
                deviation = abs(num_bars - round(num_bars))

                # Check if deviation is within tolerance (relative to one bar)
                if deviation > BAR_LENGTH_TOLERANCE:
                    messages.append(f"❌ Bar Length: '{filename}' is not a whole number of bars ({num_bars:.3f} bars at {bpm} BPM)")
                    add_issue(file_data, 'audio_bar_length')
                    has_length_issue = True

            except Exception as e:
                messages.append(f"⚠️ Error checking bar length for '{filename}': {e}")
                add_issue(file_data, 'audio_error_length')

    if not has_length_issue and checked_files > 0:
        messages.append("✅ Sample lengths match whole bars")
    elif checked_files == 0:
         messages.append("❔ No files with BPM found to check bar lengths.")

    return updated_file_data, messages

def parse_sample_filename(filename: str, file_path: Path, index_in_group: int) -> dict:
    """Parses filename to extract group, sequence, bpm, key, and attribute type."""
    base_name = filename.rsplit('.', 1)[0] # Remove extension
    original_display_name = filename

    # --- Extract Attributes --- #
    sequence = None
    seq_match = SEQUENCE_REGEX.search(base_name)
    if seq_match:
        # Group 1: _<digits>_, Group 2: _<digits>.
        seq_str = seq_match.group(1) or seq_match.group(2)
        if seq_str:
            sequence = int(seq_str)

    bpm = None
    bpm_match = BPM_REGEX.search(base_name)
    if bpm_match:
        bpm = int(bpm_match.group(1))

    key = None
    key_match = KEY_REGEX.search(base_name)
    if key_match:
        key = key_match.group(1) # Get the captured key notation
        # Normalize common variations (e.g., b -> ♭) if needed, but KEY_ORDER handles most

    # --- Determine Group Name --- #
    # Group name is the part before the first attribute found (seq, key, or bpm)
    # Find the earliest index of any attribute identifier
    indices = []
    if seq_match: indices.append(seq_match.start()) # Start index of sequence match
    if bpm_match: indices.append(bpm_match.start()) # Start index of bpm match
    if key_match: indices.append(key_match.start()) # Start index of key match

    if indices:
        group_name_end_index = min(indices)
        # Ensure we don't cut off at a leading underscore if it's the only thing before attrib
        if group_name_end_index > 0 and base_name[group_name_end_index-1] == '_':
             group_name = base_name[:group_name_end_index-1] # Exclude trailing underscore before attr
        else:
             group_name = base_name[:group_name_end_index]
    else:
        group_name = base_name # No attributes found, group is the whole base name

    # Remove trailing spaces or underscores from group name just in case
    group_name = group_name.strip().rstrip('_')

    # --- Determine Attribute Type --- #
    attribute_type = "none"
    if key is not None and bpm is not None:
        attribute_type = "key_and_bpm"
    elif key is not None:
        attribute_type = "key_only"
    elif bpm is not None:
        attribute_type = "bpm_only"

    return {
        "path": file_path,
        "display_name": original_display_name,
        "group": group_name,
        "sequence": sequence,
        "bpm": bpm,
        "key": key,
        "attribute_type": attribute_type,
        "original_index": index_in_group # Keep track of original position
    }

def validate_sample_order(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks if audio files within groups are ordered correctly by BPM, Key, Sequence."""
    messages = []
    updated_file_data = file_display_data
    has_order_issue = False
    checked_groups = 0
    sample_infos = [] # List to hold parsed info for all relevant files

    # 1. Parse all relevant filenames
    for subfolder_key, files_in_folder in updated_file_data.items():
        for idx, file_data in enumerate(files_in_folder):
            f_path = file_data['path']
            # Only parse WAV files, exclude demos
            if f_path.suffix.lower() == ".wav" and DEMO_SUFFIX.lower() not in f_path.name.lower():
                 parsed_info = parse_sample_filename(f_path.name, f_path, idx)
                 # Associate parsed info back to the original file_data dictionary
                 file_data['parsed_info'] = parsed_info
                 sample_infos.append(file_data)

    if not sample_infos:
        messages.append("❔ No WAV files found to check sample order.")
        return updated_file_data, messages

    # 2. Group samples by calculated group name and attribute type
    # We use defaultdict for easier grouping
    grouped_samples = defaultdict(list)
    for file_data in sample_infos:
        parsed = file_data['parsed_info']
        group_key = f"{parsed['group']}_{parsed['attribute_type']}"
        grouped_samples[group_key].append(file_data)

    # 3. Validate each group
    for group_key, group_file_list in grouped_samples.items():
        checked_groups += 1
        group_name_display = group_file_list[0]['parsed_info']['group'] # For display
        attribute_type_display = group_file_list[0]['parsed_info']['attribute_type'] # For display

        # Check for sequence duplicates or gaps within the group
        sequences = [fd['parsed_info']['sequence'] for fd in group_file_list if fd['parsed_info']['sequence'] is not None]
        if sequences:
            unique_sequences = set(sequences)
            if len(sequences) != len(unique_sequences):
                messages.append(f"❌ Sequence Duplicates: Group '{group_name_display}' ({attribute_type_display})")
                has_order_issue = True
                for fd in group_file_list: add_issue(fd, 'sample_order_duplicate_seq')
            else:
                min_seq, max_seq = min(sequences), max(sequences)
                # Check for gaps only if sequence starts from 1 or 0 (common patterns)
                if (min_seq == 1 and set(range(1, max_seq + 1)) != unique_sequences) or \
                   (min_seq == 0 and set(range(0, max_seq + 1)) != unique_sequences):
                    messages.append(f"❌ Sequence Gaps: Group '{group_name_display}' ({attribute_type_display})")
                    has_order_issue = True
                    for fd in group_file_list: add_issue(fd, 'sample_order_gap_seq')

        # Define the sorting key function based on attribute type
        def get_sort_key(file_data):
            parsed = file_data['parsed_info']
            bpm = parsed['bpm'] or float('inf') # Sort files without BPM last
            # Use the key name string directly for alphabetical sorting, handle None
            key_str = parsed['key'] if parsed['key'] else "~" # Puts None last alphabetically
            # key_val = KEY_ORDER.get(parsed['key'], float('inf')) if parsed['key'] else float('inf') # Old numeric key order
            seq = parsed['sequence'] or float('inf') # Sort files without sequence last

            if parsed['attribute_type'] == "key_and_bpm":
                # Sort by BPM, then Key Name (alphabetical), then Sequence
                return (bpm, key_str, seq)
            elif parsed['attribute_type'] == "key_only":
                # Sort by Key Name (alphabetical), then Sequence
                return (key_str, seq)
            elif parsed['attribute_type'] == "bpm_only":
                # Sort by BPM, then Sequence
                return (bpm, seq)
            else: # "none"
                # Sort by Sequence only
                return (seq,)

        # Determine the expected order by sorting the group
        try:
            expected_sorted_list = sorted(group_file_list, key=get_sort_key)
        except TypeError as e:
             messages.append(f"⚠️ Error sorting group '{group_name_display}' ({attribute_type_display}): {e}")
             continue # Skip comparison for this group

        # Get the current order based on original file positions within the folder
        # The `files_in_folder` list used to build `sample_infos` is already sorted naturally
        # So, we just need to compare the `expected_sorted_list` paths to the `group_file_list` paths
        # Re-fetch current order based on original index just to be safe
        current_ordered_list = sorted(group_file_list, key=lambda fd: fd['parsed_info']['original_index'])
        current_paths = [fd['path'] for fd in current_ordered_list]
        expected_paths = [fd['path'] for fd in expected_sorted_list]

        if current_paths != expected_paths:
            messages.append(f"❌ Sample Order Issue: Group '{group_name_display}' ({attribute_type_display})")
            # Provide more detailed feedback
            current_names = [p.name for p in current_paths]
            expected_names = [p.name for p in expected_paths]
            messages.append(f"    Current Order : {current_names}")
            messages.append(f"    Expected Order: {expected_names} (Sort: BPM -> Alpha Key -> Seq)")
            has_order_issue = True
            for fd in group_file_list: add_issue(fd, 'sample_order_incorrect')

    if not has_order_issue and checked_groups > 0:
        messages.append("✅ Sample order correct for all groups")

    # Clean up temporary parsed info if desired (optional)
    # for sf_key, files in updated_file_data.items():
    #     for file_data in files:
    #         if 'parsed_info' in file_data: del file_data['parsed_info']

    return updated_file_data, messages

def extract_words_for_spellcheck(filename: str) -> list[str]:
    """Extracts potential words from a filename for spell checking, ignoring technical parts and the first segment (Catalog ID)."""
    # 1. Remove extension
    base_name = filename.rsplit('.', 1)[0]

    # 2. Split into parts based on underscores
    parts = base_name.split('_')

    words_to_check = []
    # 3. Iterate through parts, *skipping the first part (Catalog ID)*
    for part in parts[1:]:
        if not part:
            continue

        # 4. Skip parts that are purely numeric, or look like key/bpm
        part_lower = part.lower()
        if part.isdigit():
            continue
        if part_lower.startswith('key') and KEY_REGEX.match(f"_{part}"):
            continue
        if part_lower.endswith('bpm') and part_lower[:-3].isdigit() and BPM_REGEX.match(f"_{part}"):
             continue
        if part_lower == "demo" or part_lower == "artwork":
             continue

        # 5. For remaining parts, split by CamelCase etc. and add to list
        potential_words_in_part = SPELLCHECK_SPLIT_WORDS_PATTERN.findall(part)
        words_to_check.extend(potential_words_in_part)

    # 6. Lowercase and filter out short words/numbers only from the collected words
    final_words = [word.lower() for word in words_to_check if len(word) > 2 and not word.isdigit()]
    # print(f"Filename: {filename}, Parts: {parts}, Final words for check: {final_words}")
    return final_words

def validate_spelling(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks filenames for potential spelling errors."""
    messages = []
    updated_file_data = file_display_data
    has_spelling_issue = False
    # Initialize SpellChecker (loads default English dictionary)
    spell = SpellChecker()
    spell.word_frequency.load_words([w.lower() for w in SPELLCHECK_IGNORE_WORDS]) # Add ignored words

    checked_files_count = 0
    for subfolder, files in updated_file_data.items():
        for file_data in files:
            checked_files_count += 1
            filename = file_data['display_name']
            words_to_check = extract_words_for_spellcheck(filename)

            if not words_to_check:
                continue

            # Find unknown words
            misspelled = spell.unknown(words_to_check)

            if misspelled:
                # Report only the first misspelled word found for brevity
                first_misspelled = list(misspelled)[0]
                messages.append(f"❌ Spelling? '{filename}' (potential issue: '{first_misspelled}')")
                add_issue(file_data, 'spelling')
                # Store the potentially misspelled word for UI if needed
                file_data['spelling_details'] = first_misspelled
                has_spelling_issue = True

    if not has_spelling_issue and checked_files_count > 0:
        messages.append("✅ Spelling seems OK")
    elif checked_files_count == 0:
         messages.append("❔ No files checked for spelling.")

    return updated_file_data, messages

def calculate_integrated_loudness(file_path: Path) -> float | None:
    """Calculates the integrated loudness (I) in LUFS for an audio file."""
    try:
        data, rate = sf.read(file_path)
        # Ensure data is float32, as pyloudnorm expects
        if data.dtype != np.float32:
            # Find max value for normalization based on dtype
            if data.dtype == np.int16:
                max_val = np.iinfo(np.int16).max
            elif data.dtype == np.int32:
                max_val = np.iinfo(np.int32).max
            elif data.dtype == np.uint8:
                 # pydub loads uint8 as centered around 128
                 data = data.astype(np.float32) - 128.0
                 max_val = 128.0
            else:
                 # Assume float64 or other float, normalize directly
                 max_val = np.max(np.abs(data))
                 if max_val == 0: return -np.inf # Silence

            if data.dtype != np.uint8: # Avoid re-normalizing uint8
                 data = data.astype(np.float32) / max_val

        meter = pyln.Meter(rate) # create BS.1770 meter
        loudness = meter.integrated_loudness(data) # measure loudness

        # Check for -inf dB loudness
        if loudness == -np.inf:
            return -np.inf # Represent silence consistently
        return float(loudness)
    except Exception as e:
        print(f"Error calculating LUFS for {file_path.name}: {e}")
        return None # Indicate error

# --- Main Orchestration Function ---
def run_all_qc_checks(file_display_data: dict) -> tuple[dict, list[str], list[dict]]:
    """Runs all QC checks and returns updated data, messages, and loudness results."""
    all_messages = []
    updated_data = file_display_data
    loudness_results = []

    st.write("Running Folder/Name Checks...")
    # ... (folder, space, demo checks)
    folder_messages = validate_folder_names(list(updated_data.keys()))
    all_messages.extend(folder_messages)
    updated_data, space_messages = validate_filename_spaces(updated_data)
    all_messages.extend(space_messages)
    updated_data, demo_messages = validate_demo_file(updated_data)
    all_messages.extend(demo_messages)

    st.write("Running Artwork Checks...")
    # ... (artwork checks)
    updated_data, artwork_messages = validate_artwork(updated_data)
    all_messages.extend(artwork_messages)

    st.write("Running Audio Spec/Analysis Checks...")
    # ... (audio specs, clipping, silence, zero crossing, length)
    updated_data, audio_spec_messages = validate_audio_specs(updated_data)
    all_messages.extend(audio_spec_messages)
    updated_data, clipping_messages = validate_clipping(updated_data)
    all_messages.extend(clipping_messages)
    updated_data, silence_messages = validate_silence(updated_data)
    all_messages.extend(silence_messages)
    updated_data, crossing_messages = validate_zero_crossings(updated_data)
    all_messages.extend(crossing_messages)
    updated_data, length_messages = validate_sample_lengths(updated_data)
    all_messages.extend(length_messages)

    st.write("Running Sample Order Check...")
    # 10. Sample Order Check
    updated_data, order_messages = validate_sample_order(updated_data)
    all_messages.extend(order_messages)

    st.write("Running Spelling Check...")
    # 11. Spelling Check
    updated_data, spelling_messages = validate_spelling(updated_data)
    all_messages.extend(spelling_messages)

    st.write("Calculating Loudness (LUFS)...")
    # 12. Loudness Calculation (after other checks)
    files_for_loudness = []
    for subfolder, files in updated_data.items():
        for file_data in files:
             # Only calculate for valid WAV files that didn't cause read errors earlier
            if file_data['path'].suffix.lower() == ".wav" and not any(e.startswith('audio_error') for e in file_data['issues']):
                files_for_loudness.append(file_data['path'])

    if files_for_loudness:
        # Use st.progress for LUFS calculation as it can be slow
        lufs_progress = st.progress(0.0, text="Calculating LUFS...")
        total_files = len(files_for_loudness)
        for i, f_path in enumerate(files_for_loudness):
            lufs_value = calculate_integrated_loudness(f_path)
            result_entry = {"Filename": f_path.name}
            if lufs_value is None:
                result_entry["Integrated Loudness (LUFS)"] = "Error"
            elif lufs_value == -np.inf:
                 result_entry["Integrated Loudness (LUFS)"] = "Silent (-inf)"
            else:
                result_entry["Integrated Loudness (LUFS)"] = f"{lufs_value:.1f}"
            loudness_results.append(result_entry)
            # Update progress bar
            progress_percent = (i + 1) / total_files
            lufs_progress.progress(progress_percent, text=f"Calculating LUFS for {f_path.name} ({i+1}/{total_files})")
        lufs_progress.empty() # Clear progress bar
    else:
         st.write("No suitable WAV files found for LUFS calculation.")

    st.write("QC Checks Complete.")

    return updated_data, all_messages, loudness_results

# --- UI Layout & Processing ---
# st.title("Make Scott Proud!") # Replaced with markdown for styling
st.markdown("<h1 style='text-align: center;'><b>MAKE SCOTT PROUD!</b></h1>", unsafe_allow_html=True)
st.markdown("---")
col1, col2 = st.columns([1, 1])

# Function to process dropped files (using st.file_uploader objects)
def process_dropped_files(uploaded_files):
    st.session_state.messages = []
    st.session_state.file_display_data = {}
    st.session_state.processing_complete = False
    st.session_state.loudness_results = []
    all_files_flat = []
    temp_dir = None

    with st.spinner('Processing uploaded files...'):
        if uploaded_files:
            try:
                # Create a temporary directory to store uploaded files
                temp_dir = tempfile.mkdtemp()
                temp_dir_path = Path(temp_dir)
                saved_file_paths = []

                # Save uploaded files to the temporary directory
                for uploaded_file in uploaded_files:
                    file_path = temp_dir_path / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_file_paths.append(file_path)

                # Now process the saved files using their paths
                parent_folders = {}
                for item_path in saved_file_paths:
                    # In this context, each saved file is treated as a top-level item
                    # We need to handle potential dropped folders differently if st.file_uploader could accept them
                    # However, st.file_uploader typically uploads individual files.
                    # If users select multiple files from one folder, they appear individually.
                    # We will group by the parent of the *saved* temp file for structure.
                    # This might differ from original structure if multiple folders selected.
                    # A more robust solution might require analyzing original paths if available, or asking user for structure.
                    # --- Simplified Grouping for now --- #
                    # Group by immediate parent (which is the temp dir) - less useful
                    # Let's try grouping by original filename parts if possible, assuming a structure like 'Folder/File.wav'
                    # This is heuristic!
                    try:
                         # Attempt to reconstruct a semblence of original structure if path has slashes
                         original_parts = Path(uploaded_file.name).parts
                         if len(original_parts) > 1:
                             # Assume the part before the last is the folder name
                             grouping_key_name = original_parts[-2]
                         else:
                             grouping_key_name = "_root_files_"
                    except Exception:
                         grouping_key_name = "_root_files_"

                    grouping_key_path = Path(grouping_key_name) # Use Path for sorting key consistency

                    # We need *all* files, not just the top-level ones for QC
                    # get_files_from_item is less relevant here as we have individual files
                    # Let's assume item_path IS the file to check
                    f_path = item_path # The actual file path in temp dir

                    if grouping_key_path not in parent_folders:
                        parent_folders[grouping_key_path] = []
                    # Store the temporary path for processing
                    parent_folders[grouping_key_path].append(f_path)

                # Sort parent folders (heuristic names) and files within them naturally
                sorted_parents = sorted(parent_folders.keys(), key=natural_sort_key)

                processed_data = {}
                for parent_key in sorted_parents:
                    sorted_files = sorted(parent_folders[parent_key], key=natural_sort_key)
                    processed_data[parent_key] = [
                        {"path": f, "display_name": f.name, "issues": []} # Use temp path, but original name
                        for f in sorted_files
                    ]
                    all_files_flat.extend(sorted_files)

                st.session_state.file_display_data = processed_data
                st.session_state.all_files_flat_paths = all_files_flat

                # Run QC Checks on the files in the temporary directory
                if st.session_state.file_display_data:
                    with st.spinner('Running QC Checks...'):
                        updated_data, qc_messages, loudness_data = run_all_qc_checks(st.session_state.file_display_data)
                        st.session_state.file_display_data = updated_data
                        st.session_state.messages = qc_messages
                        st.session_state.loudness_results = loudness_data
                else:
                    st.session_state.messages = ["No files could be processed."]

            except Exception as e:
                 st.error(f"Error during file processing: {e}")
                 st.session_state.messages = [f"❌ Error processing files: {e}"]
            finally:
                # Clean up the temporary directory
                if temp_dir:
                    try:
                        shutil.rmtree(temp_dir)
                        print(f"Removed temp directory: {temp_dir}")
                    except Exception as e:
                         print(f"Error removing temp directory {temp_dir}: {e}")

        else: # No files uploaded
            st.session_state.messages = ["No files uploaded."]

    st.session_state.processing_complete = True
    st.rerun()

with col1:
    st.header("Upload Package Folder")

    # Use st.file_uploader
    uploaded_files = st.file_uploader(
        "Drag and drop the whole package folder here",
        accept_multiple_files=True,
        type=["wav", "jpg", "jpeg"], # Specify allowed types
        key="file_uploader", # Assign a key
        on_change=lambda: setattr(st.session_state, 'reset_app', False) # Reset flag on new upload
    )

    # Button to trigger processing after files are selected
    if uploaded_files and not st.session_state.processing_complete:
         if st.button("Process Uploaded Files"):
            st.session_state.dropped_items = uploaded_files # Store uploaded file objects
            process_dropped_files(st.session_state.dropped_items)

    # Display File List (after processing)
    if st.session_state.processing_complete and not st.session_state.reset_app:
        file_list_container = st.container()
        with file_list_container:
            if not st.session_state.file_display_data:
                st.write("*(No files processed yet)*")
            else:
                st.write("**Processed Files:**")
                for parent_folder, files_in_folder in st.session_state.file_display_data.items():
                    display_parent_name = parent_folder.name # Now uses heuristic name
                    if display_parent_name == "_root_files_": display_parent_name = "(Root Files)"
                    st.markdown(f"**📁 {display_parent_name}/**")
                    for file_data in files_in_folder:
                        issues = file_data['issues']
                        color = "green"
                        prefix = "📄" # Default
                        is_artwork = file_data['path'].suffix.lower() in VALID_ARTWORK_EXT
                        # Check audio based on original name stored in display_name
                        is_audio = Path(file_data['display_name']).suffix.lower() in VALID_AUDIO_EXT

                        if is_artwork:
                            prefix = "🖼️"
                        elif is_audio:
                             prefix = "🎵"

                        if issues:
                             # Determine worst issue for color/icon (more refined)
                             has_error = any(i.startswith('audio_error') or i.startswith('artwork_error') for i in issues)
                             has_spec_issue = any(i in ['audio_format', 'audio_samplerate', 'audio_bitdepth', 'audio_channels', 'audio_clipping', 'audio_silence', 'audio_zero_crossing', 'audio_bar_length'] for i in issues)
                             has_art_issue = any(i in ['artwork_format', 'artwork_size', 'artwork_dims', 'artwork_square'] for i in issues)
                             has_order_issue = any(i.startswith('sample_order') for i in issues)
                             has_naming_issue = any(i in ['artwork_naming', 'demo_naming', 'demo_multiple'] for i in issues)
                             has_spelling_issue = 'spelling' in issues # Check for spelling issue
                             has_spacing_issue = 'spacing' in issues
                             has_potential_demo = 'potential_demo_extra' in issues

                             if has_error:
                                 color = "red"
                                 prefix = "💥" # Error reading file
                             elif has_spec_issue or has_art_issue:
                                 color = "red"
                                 prefix = "❌" # Spec violation
                             elif has_order_issue:
                                 color = "purple" # Order issues distinct color
                                 prefix = "🔢"
                             elif has_naming_issue:
                                 color = "orange"
                                 prefix = "❓" # Naming/Demo issues
                             elif has_spelling_issue:
                                 color = "#FFC300" # Darker Yellow / Gold for spelling
                                 prefix = "🔡"
                             elif has_spacing_issue or has_potential_demo:
                                 color = "yellow"
                                 prefix = "⚠️" # Minor issues
                             else:
                                 color = "grey"
                                 prefix = "❔" # Unknown issue type

                        display_text = file_data['display_name']
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='color:{color};'>{prefix} {display_text}</span>", unsafe_allow_html=True)

                st.markdown("---")
    elif not uploaded_files and not st.session_state.processing_complete:
         st.info("Upload files using the browser above.")


    # --- Buttons --- #
    st.markdown("---")
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    sorted_msgs = get_sorted_messages(st.session_state.messages)
    results_text = "\n".join(sorted_msgs)
    with btn_col1:
        if st.button("📋 COPY RESULTS", disabled=(not st.session_state.messages)):
            try:
                pyperclip.copy(results_text)
                st.toast("Results copied to clipboard!", icon="📋")
            except Exception as e:
                st.toast(f"Error copying: {e}", icon="❌")
    with btn_col2:
        st.download_button(
            label="⬆️ EXPORT RESULTS", data=results_text, file_name="QC_Results.txt",
            mime="text/plain", disabled=(not st.session_state.messages)
        )
    with btn_col3:
        if st.button("✨ RESET"):
            # Clear state, including uploader state by rerunning
            st.session_state.dropped_items = None
            st.session_state.messages = []
            st.session_state.file_display_data = {}
            st.session_state.processing_complete = False
            st.session_state.reset_app = True # This might not be needed now
            st.session_state.loudness_results = []
            if 'all_files_flat_paths' in st.session_state: del st.session_state.all_files_flat_paths
            # Clear the file uploader widget state requires a more complex approach or relying on rerun
            # st.session_state.file_uploader = [] # This might work depending on Streamlit version
            st.rerun()

with col2:
    st.header("QC Results")
    results_container = st.container()
    with results_container:
        if not st.session_state.messages:
            st.write("*(Results will appear here after processing)*")
        else:
            # Display sorted messages (using the already sorted list)
            for msg in sorted_msgs:
                st.markdown(msg)

# --- Loudness Table Display (Below Columns) --- #
st.markdown("---")
st.header("Loudness Analysis (Integrated LUFS)")
loudness_df = None
if st.session_state.processing_complete and st.session_state.loudness_results:
    # Create DataFrame for display
    loudness_df = pd.DataFrame(st.session_state.loudness_results)
    st.dataframe(loudness_df, use_container_width=True)

    # --- Find Loudest/Quietest --- #
    valid_lufs_entries = []
    for entry in st.session_state.loudness_results:
        lufs_str = entry["Integrated Loudness (LUFS)"]
        filename = entry["Filename"]
        try:
            # Handle "Silent (-inf)" specifically
            if isinstance(lufs_str, str) and "-inf" in lufs_str:
                lufs_float = -np.inf
            else:
                lufs_float = float(lufs_str) # Convert valid numbers
            valid_lufs_entries.append({"filename": filename, "lufs": lufs_float})
        except (ValueError, TypeError):
            continue # Skip "Error" or other non-convertible entries

    loudest_file = None
    quietest_file = None
    max_lufs = -np.inf
    min_lufs = np.inf

    if valid_lufs_entries:
        for entry in valid_lufs_entries:
            # Update loudest
            if entry["lufs"] > max_lufs:
                max_lufs = entry["lufs"]
                loudest_file = entry
            # Update quietest (handle -inf correctly)
            if entry["lufs"] < min_lufs:
                min_lufs = entry["lufs"]
                quietest_file = entry

        st.markdown("**Loudness Extremes:**")
        col_loud, col_quiet = st.columns(2)
        with col_loud:
            if loudest_file:
                 lufs_display = "Silent" if loudest_file['lufs'] == -np.inf else f"{loudest_file['lufs']:.1f} LUFS"
                 st.metric(label="🔊 Loudest File", value=lufs_display, delta=loudest_file['filename'])
            else:
                st.info("No valid loudness data for loudest file.")
        with col_quiet:
             if quietest_file:
                 lufs_display = "Silent" if quietest_file['lufs'] == -np.inf else f"{quietest_file['lufs']:.1f} LUFS"
                 st.metric(label="🤫 Quietest File", value=lufs_display, delta=quietest_file['filename'])
             else:
                 st.info("No valid loudness data for quietest file.")

elif st.session_state.processing_complete:
    st.info("No suitable WAV files found or processed for loudness analysis.")
else:
    st.info("Process files to see loudness analysis.")

# --- Initial State Display Logic ---
# Handled within component display logic 
