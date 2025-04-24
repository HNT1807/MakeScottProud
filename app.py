import streamlit as st
import os
import re
from pathlib import Path
import natsort
import time
import numpy as np
from PIL import Image
from pydub import AudioSegment
from pydub.utils import mediainfo
from collections import defaultdict
from spellchecker import SpellChecker
import pyperclip
import soundfile as sf
import pyloudnorm as pyln
import pandas as pd
import shutil
import tempfile
import zipfile
from io import StringIO

# --- Constants ---
VALID_SUBFOLDERS = {"Loops", "One Shots"}
VALID_EXTENSIONS = {".wav", ".jpg", ".jpeg"}
DEMO_SUFFIX = "_Demo.wav"
ARTWORK_SUFFIX = "_Artwork"
VALID_ARTWORK_EXT = {".jpg", ".jpeg"}
TARGET_ARTWORK_EXT = ".jpg"
MAX_ARTWORK_SIZE_MB = 4
MIN_ARTWORK_DIM = 1000
VALID_AUDIO_EXT = {".wav"}
TARGET_SAMPLE_RATE = 44100
TARGET_BIT_DEPTH = 24
TARGET_CHANNELS = 2
SILENCE_THRESHOLD_DBFS = -60.0
ZERO_CROSSING_THRESHOLD = 0.001
BAR_LENGTH_TOLERANCE = 0.001
BPM_REGEX = re.compile(r"_(\d+)bpm")
SEQUENCE_REGEX = re.compile(r"_(\d+)_|_(\d+)\.(?=[^.]*$)")
KEY_REGEX = re.compile(r"_key([A-Ga-g][#b♭]?[mM]?)")
KEY_ORDER = {
    "C": 0, "Cm": 0.1, "C#": 1, "Db": 1, "C#m": 1.1, "Dbm": 1.1,
    "D": 2, "Dm": 2.1, "D#": 3, "Eb": 3, "D#m": 3.1, "Ebm": 3.1,
    "E": 4, "Em": 4.1, "F": 5, "Fm": 5.1, "F#": 6, "Gb": 6, "F#m": 6.1, "Gbm": 6.1,
    "G": 7, "Gm": 7.1, "G#": 8, "Ab": 8, "G#m": 8.1, "Abm": 8.1,
    "A": 9, "Am": 9.1, "A#": 10, "Bb": 10, "A#m": 10.1, "Bbm": 10.1,
    "B": 11, "Bm": 11.1
}
SPELLCHECK_IGNORE_WORDS = {"Metastarter", "Cymatics", "Ableton", "Synth", "Serum", "Vital", "Phaseplant"}
SPELLCHECK_REMOVE_PATTERNS = re.compile(r"_key[A-Ga-g][#b♭]?[mM]?|_\d+bpm|_\d+_|_\d+$|_+Demo$|_Artwork$", re.IGNORECASE)
SPELLCHECK_SPLIT_WORDS_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)|\d+")

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Make Scott Proud!")

# --- State Management ---
# Remove folder_path text input state
# if 'folder_path' not in st.session_state:
#     st.session_state.folder_path = ""
if 'uploaded_zip_path' not in st.session_state:
    st.session_state.uploaded_zip_path = None
if 'extracted_temp_dir' not in st.session_state:
    st.session_state.extracted_temp_dir = None # To store path for cleanup
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_display_data' not in st.session_state:
    st.session_state.file_display_data = {}
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'loudness_results' not in st.session_state:
    st.session_state.loudness_results = []

# --- Helper Functions ---
def natural_sort_key(path):
    """Provides a key for natural sorting of Path objects."""
    return natsort.natsort_keygen()(path.name)

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
    """Sorts messages by icon prefix and adds blank lines between groups."""
    # Define the desired order of icons
    icon_order = [
        "✅", "❔", "💥", "❌",
        "📎", # Added Clipping icon
        "📏", "🔈", "👂", "🔢", "❓", "��", "⚠️"
    ]
    # Use defaultdict to group messages by their first character (icon)
    grouped_messages = defaultdict(list)
    other_messages = [] # For messages with unknown/no icon

    for msg in messages:
        found_prefix = False
        for icon in icon_order:
            if msg.startswith(icon):
                grouped_messages[icon].append(msg)
                found_prefix = True
                break
        if not found_prefix:
            other_messages.append(msg)

    # Build the final list in the specified order, adding separators
    final_messages = []
    for icon in icon_order:
        if icon in grouped_messages:
            # Add separator if this is not the first group being added
            if final_messages:
                final_messages.append("")
            # Add sorted messages for this icon group
            final_messages.extend(sorted(grouped_messages[icon]))

    # Add any messages with unknown/no prefixes at the end
    if other_messages:
        if final_messages:
            final_messages.append("")
        final_messages.extend(sorted(other_messages))

    return final_messages

# --- QC Check Functions ---
# Define validate_folder_names to work with actual subdirs
def validate_folder_names(file_display_data: dict) -> list[str]:
    """Checks if any found subfolders have names other than 'Loops' or 'One Shots'."""
    messages = []
    # required_missing = [] # Removed
    invalid_found = []
    
    # Get the names of the subfolders actually found and processed
    # Exclude the special key for root files
    found_subfolder_names = {key for key in file_display_data if key != "_root_files_"}

    # # Check for missing required folders # Removed
    # for req_folder in VALID_SUBFOLDERS:
    #     if req_folder not in found_subfolder_names:
    #         required_missing.append(req_folder)

    # Check for unexpected folders found among the processed ones
    for found_folder in found_subfolder_names:
         if found_folder not in VALID_SUBFOLDERS:
             invalid_found.append(found_folder)

    # Report error only if invalid folders were found
    if invalid_found:
        messages.append(f"❔ Unexpected subfolders found: {invalid_found}. Only '{ '/'.join(VALID_SUBFOLDERS) }' allowed if subfolders exist.")
    else:
        # Success message if no invalid folders found (even if no subfolders exist)
        messages.append(f"✅ Subfolder names are valid (if any subfolders exist). Allowed: '{ '/'.join(VALID_SUBFOLDERS) }'")

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
            messages.append(f"❔ Missing Demo file or mispelled one (must end with '{DEMO_SUFFIX}')")
            for file_data in potential_demos:
                add_issue(file_data, 'demo_missing_suffix')
        else:
            messages.append(f"❔ Missing Demo file or mispelled one (must end with '{DEMO_SUFFIX}')")

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
        messages.append(f"❔ Missing Artwork file or mispelled one (must end with '{ARTWORK_SUFFIX}.jpg')")
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
        messages.append("✅ Audio file specs correct")
    elif checked_files == 0 and not found_non_wav:
        messages.append("❔ No .wav files found to check specs.")

    return updated_file_data, messages

def validate_clipping(file_display_data: dict) -> tuple[dict, list[str]]:
    """Checks audio files for clipping (peaks >= 0.999)."""
    messages = []
    updated_file_data = file_display_data
    has_clipping = False
    checked_files = 0
    CLIP_THRESHOLD = 0.999 # Define threshold slightly below 1.0
    for subfolder, files in updated_file_data.items():
        for file_data in files:
            f_path = file_data['path']
            if f_path.suffix.lower() not in VALID_AUDIO_EXT:
                continue
            checked_files += 1
            try:
                audio = AudioSegment.from_file(str(f_path))
                samples = get_normalized_samples(audio)
                if samples is not None and np.max(np.abs(samples)) >= CLIP_THRESHOLD:
                    messages.append(f"📎 Clipping detected in '{file_data['display_name']}')") # Changed icon
                    add_issue(file_data, 'audio_clipping')
                    has_clipping = True
            except Exception as e:
                messages.append(f"⚠️ Error checking clipping for '{file_data['display_name']}': {e}")
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
                    messages.append(f"👂 Silent track detected: '{file_data['display_name']}' ({audio.dBFS:.2f} dBFS)")
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
                    messages.append(f"🔈 Zero Crossing: '{file_data['display_name']}' does not {' and '.join(issue_desc)} on zero crossing.")
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
                    messages.append(f"📏 Bar Length: '{filename}' is not a whole number of bars ({num_bars:.3f} bars at {bpm} BPM)")
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
                messages.append(f"🔢 Sequence Duplicates: Group '{group_name_display}' ({attribute_type_display})")
                has_order_issue = True
                for fd in group_file_list: add_issue(fd, 'sample_order_duplicate_seq')
            else:
                min_seq, max_seq = min(sequences), max(sequences)
                # Check for gaps only if sequence starts from 1 or 0 (common patterns)
                if (min_seq == 1 and set(range(1, max_seq + 1)) != unique_sequences) or \
                   (min_seq == 0 and set(range(0, max_seq + 1)) != unique_sequences):
                    messages.append(f"🔢 Sequence Gaps: Group '{group_name_display}' ({attribute_type_display})")
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
            messages.append(f"🔢 Sample Order Issue: Group '{group_name_display}' ({attribute_type_display})")
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
                messages.append(f"🔡 Spelling? '{filename}' (potential issue: '{first_misspelled}')")
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
                 # Correct uint8 handling
                 data = data.astype(np.float32) - 128.0
                 max_val = 128.0
            # Check if max_val was determined (i.e., it was int or uint8)
            if 'max_val' in locals() and data.dtype != np.uint8: # Avoid re-normalizing uint8
                 data = data.astype(np.float32) / max_val
            elif 'max_val' not in locals(): # Handle other float types
                max_val = np.max(np.abs(data))
                if max_val == 0: return -np.inf # Silence
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

# --- New function to process folder path ---
def process_folder_path(folder_path_str: str):
    """Processes files directly from a given folder path, handling subdirs dynamically."""
    # Clear previous results *before* processing new path
    st.session_state.messages = [] 
    st.session_state.file_display_data = {}
    st.session_state.processing_complete = False
    st.session_state.loudness_results = []

    root_path = Path(folder_path_str)

    with st.spinner(f'Processing folder: {folder_path_str}...'):
        # Validate root path
        if not root_path.exists():
            st.error(f"Error: Folder path does not exist: {root_path}")
            st.session_state.messages = [f"❌ Error: Folder path does not exist: {root_path}"]
            st.session_state.processing_complete = True
            st.rerun()
            return # Stop processing
        if not root_path.is_dir():
            st.error(f"Error: Provided path is not a folder: {root_path}")
            st.session_state.messages = [f"❌ Error: Provided path is not a folder: {root_path}"]
            st.session_state.processing_complete = True
            st.rerun()
            return # Stop processing

        processed_data = defaultdict(list)
        root_files_list = []
        try:
            # --- Dynamic iteration over root path contents --- #
            for item in root_path.iterdir():
                # Ignore hidden files/folders
                if item.name.startswith('.'):
                    continue

                # Process Subdirectories
                if item.is_dir():
                    subfolder_path = item
                    files_in_subfolder = []
                    for sub_item in subfolder_path.iterdir():
                        # Only include files with valid extensions
                        if sub_item.is_file() and sub_item.suffix.lower() in VALID_EXTENSIONS:
                            files_in_subfolder.append(sub_item)

                    if files_in_subfolder:
                         # Sort naturally
                         sorted_files = sorted(files_in_subfolder, key=natural_sort_key)
                         # Store in the desired format using the actual subfolder name
                         processed_data[subfolder_path.name] = [
                             {
                                 "path": file_path,
                                 "display_name": file_path.name,
                                 "issues": []
                             }
                             for file_path in sorted_files
                         ]

                # Process Root Files
                elif item.is_file() and item.suffix.lower() in VALID_EXTENSIONS:
                     # Check parent ensures it's directly in root_path, not already processed above
                     if item.parent == root_path:
                          root_files_list.append(item)
            # --- End dynamic iteration --- #

            # Add collected root files (if any) after sorting
            if root_files_list:
                sorted_root_files = sorted(root_files_list, key=natural_sort_key)
                processed_data["_root_files_"] = [
                     {
                         "path": file_path,
                         "display_name": file_path.name,
                         "issues": []
                     }
                     for file_path in sorted_root_files
                ]

        except Exception as e:
            st.error(f"Error accessing or reading folder structure: {e}")
            st.session_state.messages.append(f"❌ Error reading folder: {e}")
            st.session_state.processing_complete = True # Mark as complete even on error
            st.rerun()
            return

        # Check if any valid files were found (in root or subdirs)
        if not processed_data:
            st.warning(f"No files with valid extensions ({', '.join(VALID_EXTENSIONS)}) found in the specified folder or its direct subdirectories.")
            st.session_state.messages.append("❔ No processable files found.")
            st.session_state.file_display_data = {} # Ensure it's empty
        else:
            # --- Run QC Checks --- #
            st.session_state.file_display_data = dict(processed_data) # Convert defaultdict
            if st.session_state.file_display_data:
                with st.spinner('Running QC Checks...'):
                    updated_data, qc_messages, loudness_data = run_all_qc_checks(st.session_state.file_display_data)
                    st.session_state.file_display_data = updated_data
                    st.session_state.messages.extend(qc_messages) # Extend existing messages
                    st.session_state.loudness_results = loudness_data
            else:
                 # This case might be redundant due to the outer check, but safe to keep
                 st.session_state.messages.append("No files could be processed for QC checks.")

    st.session_state.processing_complete = True
    st.rerun()

# --- Main Orchestration Function ---
def run_all_qc_checks(file_display_data: dict) -> tuple[dict, list[str], list[dict]]:
    """Runs all QC checks and returns updated data, messages, and loudness results."""
    all_messages = []
    updated_data = file_display_data
    loudness_results = []

    st.write("Running Folder Structure Check...")
    # Validate based on the keys found in file_display_data
    folder_messages = validate_folder_names(updated_data) # Pass the dictionary
    all_messages.extend(folder_messages)

    st.write("Running Filename/Demo/Artwork Checks...")
    updated_data, space_messages = validate_filename_spaces(updated_data)
    all_messages.extend(space_messages)
    updated_data, demo_messages = validate_demo_file(updated_data)
    all_messages.extend(demo_messages)
    updated_data, artwork_messages = validate_artwork(updated_data)
    all_messages.extend(artwork_messages)

    st.write("Running Audio Spec/Analysis Checks...")
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
    updated_data, order_messages = validate_sample_order(updated_data)
    all_messages.extend(order_messages)

    st.write("Running Spelling Check...")
    updated_data, spelling_messages = validate_spelling(updated_data)
    all_messages.extend(spelling_messages)

    st.write("Calculating Loudness (LUFS)...")
    # Loudness calculation
    files_for_loudness = []
    for subfolder_key, files in updated_data.items():
        for file_data in files:
            # Use the temp path for calculation
            if file_data['path'].suffix.lower() == ".wav" and not any(e.startswith('audio_error') for e in file_data['issues']):
                files_for_loudness.append(file_data['path'])
    if files_for_loudness:
        lufs_progress = st.progress(0.0, text="Calculating LUFS...")
        total_files = len(files_for_loudness)
        for i, f_path in enumerate(files_for_loudness):
            lufs_value = calculate_integrated_loudness(f_path)
            # Use original name for reporting
            original_name = "Unknown" # Find original name corresponding to temp path if needed
            # This requires linking temp path back to original name or storing it differently
            # Let's find it back in the main data structure
            for pk, file_list in updated_data.items():
                for fd in file_list:
                    if fd['path'] == f_path:
                        original_name = fd['display_name']
                        break
                if original_name != "Unknown": break

            result_entry = {"Filename": original_name}
            if lufs_value is None:
                result_entry["Integrated Loudness (LUFS)"] = "Error"
            elif lufs_value == -np.inf:
                 result_entry["Integrated Loudness (LUFS)"] = "Silent (-inf)"
            else:
                result_entry["Integrated Loudness (LUFS)"] = f"{lufs_value:.1f}"
            loudness_results.append(result_entry)
            progress_percent = (i + 1) / total_files
            lufs_progress.progress(progress_percent, text=f"Calculating LUFS for {original_name} ({i+1}/{total_files})")
        lufs_progress.empty()
    else:
         st.write("No suitable WAV files found for LUFS calculation.")

    st.write("QC Checks Complete.")
    return updated_data, all_messages, loudness_results

# --- UI Layout & Processing ---
st.markdown("<h1 style='text-align: center;'><b>MAKE SCOTT PROUD!</b></h1>", unsafe_allow_html=True)
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    # Center the header using markdown and HTML
    st.markdown("<h2 style='text-align: center;'>Upload Package Folder (ZIP)</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload the package folder as a single .zip file:",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_uploader" # Unique key for the uploader
    )

    # Remove old text input and button logic
    # st.session_state.folder_path = st.text_input(...)
    # process_button_placeholder = st.empty()
    # if st.session_state.folder_path and not st.session_state.processing_complete:
    #    if process_button_placeholder.button("Process Folder Path"):
    #        process_folder_path(st.session_state.folder_path)

    # --- Extraction and Processing Logic (NEW) --- 
    if uploaded_file is not None:
        # Check if we've already processed this specific upload
        # This prevents re-processing on every script rerun after upload
        if st.session_state.uploaded_zip_path != uploaded_file.file_id:
            
            # Clear previous potential temp dir before creating new one
            if st.session_state.extracted_temp_dir:
                print(f"Attempting cleanup of old temp dir: {st.session_state.extracted_temp_dir}")
                shutil.rmtree(st.session_state.extracted_temp_dir, ignore_errors=True)
                st.session_state.extracted_temp_dir = None

            st.session_state.uploaded_zip_path = uploaded_file.file_id # Store file_id
            st.session_state.processing_complete = False # Reset processing status
            st.session_state.messages = ["🏁 Starting processing..."] # Initial message
            st.session_state.file_display_data = {}
            st.session_state.loudness_results = []

            progress_bar = st.progress(0, text="Extracting ZIP file...")
            try:
                # Create a temporary directory
                temp_dir = tempfile.mkdtemp()
                st.session_state.extracted_temp_dir = temp_dir # Store for cleanup
                temp_dir_path = Path(temp_dir)
                print(f"Created temp dir: {temp_dir}")

                # Extract the zip file
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir_path)
                progress_bar.progress(0.5, text="ZIP extracted. Finding content root...")
                print(f"Extracted zip to: {temp_dir}")

                # --- Determine the actual content root --- #
                # List items in the temp dir
                extracted_items = list(temp_dir_path.iterdir())
                # Filter out macOS metadata folders/files if they exist
                extracted_items = [item for item in extracted_items if item.name != '__MACOSX' and not item.name.startswith('._')]
                
                content_root_path = temp_dir_path
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    # If there's exactly one item and it's a directory, assume it's the root
                    content_root_path = extracted_items[0]
                    print(f"Single root folder found: {content_root_path.name}. Using it as content root.")
                else:
                    print("Multiple items or no single root folder found. Using temp dir as content root.")

                progress_bar.progress(0.7, text="Starting QC checks...")

                # Trigger processing with the determined content root path
                # Ensure process_folder_path clears previous results internally
                process_folder_path(str(content_root_path))
                # No need to call rerun here, process_folder_path ends with rerun

            except zipfile.BadZipFile:
                st.error("Error: Uploaded file is not a valid ZIP archive.")
                st.session_state.messages = ["❌ Error: Invalid ZIP file."]
                st.session_state.processing_complete = True
                if st.session_state.extracted_temp_dir:
                    shutil.rmtree(st.session_state.extracted_temp_dir, ignore_errors=True)
                    st.session_state.extracted_temp_dir = None
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during extraction or processing: {e}")
                st.session_state.messages = [f"❌ Error: {e}"]
                st.session_state.processing_complete = True
                if st.session_state.extracted_temp_dir:
                    shutil.rmtree(st.session_state.extracted_temp_dir, ignore_errors=True)
                    st.session_state.extracted_temp_dir = None
                st.rerun()
            finally:
                # Ensure progress bar is removed
                progress_bar.empty()
        # else: # Optional: uncomment to see when it skips reprocessing
        #     print("Skipping re-processing of already uploaded file.")
        
    # --- File List Display (Restored) ---
    if st.session_state.processing_complete and st.session_state.file_display_data:
        st.markdown("---")
        st.subheader("Processed Files")
        # Iterate through subfolders and files
        for subfolder, files in st.session_state.file_display_data.items():
            # Handle root files display name
            display_folder_name = "Root Folder Files" if subfolder == "_root_files_" else subfolder
            with st.expander(f"**{display_folder_name}** ({len(files)} files)", expanded=True):
                for file_data in files:
                    display_name = file_data['display_name']
                    issues = file_data['issues']
                    icon = "✅"
                    color = "#4CAF50" # Green
                    tooltip_text = "No issues detected."
                    # Simplified issue prioritization for example
                    if issues:
                        issue_str = ", ".join(issues)
                        # Prioritize Errors first
                        error_keywords = ['error', 'format', 'missing', 'spacing', 'artwork_naming', 'failed', 'duplicate', 'incorrect', 'order'] # Added order
                        warning_keywords = ['clipping', 'length', 'crossing', 'samplerate', 'bitdepth', 'channels', 'size', 'dims', 'square', 'naming', 'spelling', 'gap', 'demo_missing_suffix'] # Added demo suffix
                        info_keywords = ['silence'] 
                        
                        if any(kw in issue for issue in issues for kw in error_keywords):
                            icon = "❌"
                            color = "#F44336" # Red
                            tooltip_text = f"Errors: {issue_str}"
                        elif any(kw in issue for issue in issues for kw in warning_keywords):
                            icon = "⚠️"
                            color = "#FFC107" # Amber
                            tooltip_text = f"Warnings: {issue_str}"
                        elif any(kw in issue for issue in issues for kw in info_keywords):
                            icon = "👂"
                            color = "#2196F3" # Blue
                            tooltip_text = f"Info: {issue_str}"
                        else: # Fallback for unclassified issues like ok markers
                            # Check for positive markers if no negative ones found
                            if 'demo_ok' in issues or 'artwork_ok' in issues:
                                icon = "✅"
                                color = "#4CAF50"
                                tooltip_text = f"Status: {issue_str}"
                            else:
                                icon = "ℹ️" # Default info for uncategorized
                                color = "grey"
                                tooltip_text = f"Info: {issue_str}"
                                
                    # Display filename with icon and color, NO individual copy button
                    st.markdown(f"<span style='color:{color};' title='{tooltip_text}'>{icon} {display_name}</span>", unsafe_allow_html=True)

    # Initial placeholder for file list area - Updated condition
    elif uploaded_file is None:
         st.info("Upload the package folder as a .zip file to begin.")
    elif not st.session_state.processing_complete and st.session_state.uploaded_zip_path is not None:
         st.info("Processing uploaded file...") # Show processing message

    # --- Buttons (Adjusted) --- #
    # Removed btn_col1, btn_col2, btn_col3 = st.columns(3)
    # Removed 'Copy All Filenames' button
    # btn_col2 is empty

    # Center the Reset button # -- REMOVING THIS SECTION --
    # _, center_reset_col, _ = st.columns([1, 1, 1]) # Use columns for centering
    # with center_reset_col:
    #     if st.button("✨ RESET", use_container_width=True): # Add use_container_width
    #         # --- Add cleanup for temp directory --- #
    #         if st.session_state.extracted_temp_dir:
    #             print(f"Reset requested. Cleaning up temp dir: {st.session_state.extracted_temp_dir}")
    #             shutil.rmtree(st.session_state.extracted_temp_dir, ignore_errors=True)
    #             st.session_state.extracted_temp_dir = None
    #         # --- End of cleanup add --- #
    #         # Clear results and internal state
    #         st.session_state.uploaded_zip_path = None # Clear uploaded file tracking
    #         st.session_state.messages = []
    #         st.session_state.file_display_data = {}
    #         st.session_state.processing_complete = False
    #         st.session_state.loudness_results = []
    #         st.rerun()
    pass # Add pass if no other elements remain directly under col1

with col2:
    # Center the header using markdown and HTML
    st.markdown("<h2 style='text-align: center;'>QC Results</h2>", unsafe_allow_html=True)
    # Add vertical space after the header
    st.markdown("<br>", unsafe_allow_html=True)
    results_container = st.container()
    sorted_msgs = [] # Define outside the else block
    txt_data = "" # Define txt_data outside the block as well
    with results_container:
        if not st.session_state.messages:
            # Center the placeholder text
            st.markdown("<p style='text-align: center; color: grey;'><i>(Results will appear here after processing)</i></p>", unsafe_allow_html=True)
            # st.write("*(Results will appear here after processing)*") # Removed original write
        else:
            # Sort messages before displaying
            sorted_msgs = get_sorted_messages(st.session_state.messages)
            # Format messages for text file/clipboard (simple join with newlines)
            txt_data = "\n".join(sorted_msgs)
            # Display sorted messages (using the now defined sorted_msgs)
            for msg in sorted_msgs:
                st.markdown(msg)
                
    # --- Add TXT Export & Copy Buttons for QC Results (Centered) --- #
    if st.session_state.processing_complete and sorted_msgs:
        # Add vertical space before buttons
        st.markdown("<br>", unsafe_allow_html=True)
        # Use columns to center the buttons vertically
        _, center_col, _ = st.columns([1, 2, 1]) # Adjust ratios if needed
        with center_col:
            # Export Button
            st.download_button(
                label="💾 Export QC Results (.txt)",
                data=txt_data,
                file_name="qc_results_report.txt",
                mime="text/plain",
                key="export_txt_button",
                use_container_width=True # Make button fill the center column
            )
            # Copy Button
            if st.button("📋 Copy QC Results", key="copy_qc_button", use_container_width=True):
                pyperclip.copy(txt_data)
                st.toast("Copied QC results to clipboard!")
    else:
        # Add vertical space before disabled buttons as well
        st.markdown("<br>", unsafe_allow_html=True)
        # Show disabled buttons, also centered
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.button("💾 Export QC Results (.txt)", disabled=True, use_container_width=True)
            st.button("📋 Copy QC Results", disabled=True, use_container_width=True)

# --- Loudness Table Display (Below Columns) --- #
st.markdown("---")
st.header("Loudness Analysis (Integrated LUFS)")
loudness_df = None
if st.session_state.processing_complete and st.session_state.loudness_results:
    # Create DataFrame for display
    loudness_df = pd.DataFrame(st.session_state.loudness_results)
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

    # Display the DataFrame Table AFTER the extremes
    st.dataframe(loudness_df, use_container_width=True)

elif st.session_state.processing_complete:
    st.info("No suitable WAV files found or processed for loudness analysis.")
else:
    st.info("Process files to see loudness analysis.")

# --- Initial State Display Logic ---
# Handled within component display logic 
