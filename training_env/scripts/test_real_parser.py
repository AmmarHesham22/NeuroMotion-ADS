import sys
import os
import torch


from preprocessing.parser import JSONParser

def main():
    print("Testing Real JSON Parser...")
    
    # Initialize parser
    parser = JSONParser(window_size=300, num_joints=17, gaze_dim=4)
    
    # Target file
    file_path = r"E:\datasets\snd1156-1-1\data\data_example.json"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    print(f"Parsing {file_path}...")
    data = parser.parse_dream_stream(file_path)
    
    skel = data["skeleton"]
    gaze = data["gaze"]
    ados = data["ados_target"]
    meta = data["metadata"]
    
    print("\n--- Parsed Tensors ---")
    print(f"Skeleton Tensor Shape: {skel.shape} -> Expected: [3, 300, 17]")
    print(f"Gaze Tensor Shape:     {gaze.shape} -> Expected: [300, 4]")
    print(f"ADOS Target Score:     {ados} (type: {type(ados)})")
    print(f"Metadata:              {meta}")
    
    # Check if we successfully extracted non-zero values
    non_zero_skel = (skel != 0).sum().item()
    non_zero_gaze = (gaze != 0).sum().item()
    
    print("\n--- Value Diagnostics ---")
    print(f"Non-zero Skeleton Elements: {non_zero_skel} / {skel.numel()}")
    print(f"Non-zero Gaze Elements:     {non_zero_gaze} / {gaze.numel()}")
    if non_zero_skel == 0:
        print("WARNING: Skeleton tensor is entirely zeros. Data may not match the schema.")
    if non_zero_gaze == 0:
        print("WARNING: Gaze tensor is entirely zeros. Data may not match the schema.")

if __name__ == "__main__":
    main()
