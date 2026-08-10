import re
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

def extract_time_range(filename: str) -> tuple[int, int]:
    """Extract start and end seconds from filename like [video_id]-[30-40].wav"""
    # Find the pattern [start-end] at the end
    match = re.search(r'\[(\d+)-(\d+)\]\.wav$', filename)
    if match:
        start_sec = int(match.group(1))
        end_sec = int(match.group(2))
        return start_sec, end_sec
    else:
        raise ValueError(f"Could not extract time range from filename: {filename}")

def clip_audio_file(input_path: Path, output_path: Path, start_sec: int, end_sec: int):
    """Load audio file, clip to specified time range, and save"""
    try:
        # Load the audio file
        audio, sr = librosa.load(input_path, sr=None)  # Keep original sample rate
        
        # Calculate start and end samples
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        
        # Clip the audio
        clipped_audio = audio[start_sample:end_sample]
        
        # Save the clipped audio
        sf.write(output_path, clipped_audio, sr)
        
        return True, f"Successfully clipped {input_path.name}"
    except Exception as e:
        return False, f"Error processing {input_path.name}: {str(e)}"

def process_audio_files(filenames_paths: list, output_dir: str):
    """Process all audio files in the list"""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    print(f"Processing {len(filenames_paths)} audio files...")
    print(f"Output directory: {output_dir}")
    
    for file_path in tqdm(filenames_paths, desc="Clipping audio files"):
        try:
            # Extract time range from filename
            start_sec, end_sec = extract_time_range(file_path.name)
            
            # Create output file path with same name
            output_file = output_path / file_path.name
            
            # Skip if file already exists
            if output_file.exists():
                print(f"Skipping {file_path.name} - already exists")
                continue
            
            # Clip and save audio
            success, message = clip_audio_file(file_path, output_file, start_sec, end_sec)
            
            if success:
                successful += 1
            else:
                failed += 1
                print(f"FAILED: {message}")

        except Exception as e:
            failed += 1
            print(f"FAILED: Error processing {file_path.name}: {str(e)}")

    print(f"\\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(filenames_paths)}")

input_unclipped_dir = "/net/storage/pr3/plgrid/plgg_dynamic/music_caps_shorted"
output_clipped_dir = "/net/storage/pr3/plgrid/plgg_dynamic/music_caps_shorted"

path_musiccaps = Path(input_unclipped_dir)
paths_songs = [p for p in path_musiccaps.glob("*.wav")]
filenames_paths = [p for p in path_musiccaps.glob("*.wav")]

process_audio_files(filenames_paths, output_clipped_dir)
