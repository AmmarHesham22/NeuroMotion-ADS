import os
import sys
import glob
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.parser import JSONParser

def process_directory(source_dir: str, output_dir: str, dataset_type: str, window_size: int = 300):
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    parser = JSONParser(window_size=window_size)
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    
    if not json_files:
        # Check subdirectories
        json_files = glob.glob(os.path.join(source_dir, "**", "*.json"), recursive=True)
        
    print(f"Found {len(json_files)} JSON files in {source_dir} for {dataset_type}.")
    
    for file_path in json_files:
        print(f"Processing {file_path}...")
        
        try:
            if dataset_type == "DREAM":
                data = parser.parse_dream_stream(file_path)
            else:
                data = parser.parse_pinsoro_stream(file_path)
                
            # data contains "skeleton" [3, T, V] and "gaze" [T, D]
            # Since the parser truncates to window_size, we just save this chunk.
            # If the file had more frames, a more advanced segmenter would slice it into multiple chunks.
            # For now, we save the first window_size frames.
            
            base_name = os.path.basename(file_path).replace(".json", "")
            out_file = os.path.join(output_dir, f"{dataset_type}_{base_name}.pt")
            
            torch.save(data, out_file)
            print(f"Saved optimized tensor to {out_file}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    print("Starting Full Dataset Compilation...")
    
    # Paths
    base_data_dir = r"E:\datasets\snd1156-1-1\data"
    dream_dir = os.path.join(base_data_dir, "dream")
    pinsoro_dir = os.path.join(base_data_dir, "pinsoro")
    
    output_dir = os.path.join(base_data_dir, "processed")
    
    # If the directories don't exist, we'll try to find any json in the root data folder for testing
    if not os.path.exists(dream_dir):
        print(f"Creating mock {dream_dir} and copying data_example.json for testing...")
        os.makedirs(dream_dir, exist_ok=True)
        example_json = os.path.join(base_data_dir, "data_example.json")
        if os.path.exists(example_json):
            import shutil
            shutil.copy(example_json, os.path.join(dream_dir, "data_example.json"))
            
    if not os.path.exists(pinsoro_dir):
        print(f"Creating empty {pinsoro_dir}...")
        os.makedirs(pinsoro_dir, exist_ok=True)
        
    process_directory(dream_dir, output_dir, dataset_type="DREAM", window_size=300)
    process_directory(pinsoro_dir, output_dir, dataset_type="PInSoRo", window_size=300)
    
    print("Dataset compilation completed.")

if __name__ == "__main__":
    main()
