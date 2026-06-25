import os
import sys
import glob
import torch
import concurrent.futures

# إضافة المسار الرئيسي عشان يقدر يقرا باقي ملفات المشروع
from preprocessing.parser import JSONParser

# دالة معالجة الملف الواحد
def process_single_file(file_path, output_dir, dataset_type, window_size):
    base_name = os.path.basename(file_path).replace(".json", "")
    out_file = os.path.join(output_dir, f"{dataset_type}_{base_name}.pt")
    
    # التأكد إذا كان الملف اتعالج قبل كده لتوفير الوقت
    if os.path.exists(out_file):
        return f"Skipped (already exists): {out_file}"
        
    parser = JSONParser(window_size=window_size)
    try:
        if dataset_type == "DREAM":
            data = parser.parse_dream_stream(file_path)
        else:
            data = parser.parse_pinsoro_stream(file_path)
            
        torch.save(data, out_file)
        return f"Saved optimized tensor to {out_file}"
    except Exception as e:
        return f"Error processing {file_path}: {e}"

# دالة معالجة الفولدر بالكامل
def process_directory(source_dir: str, output_dir: str, dataset_type: str, window_size: int = 300):
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # السطر ده هيجبره يبحث جوه كل الفولدرات الفرعية بتاعت اليوزرز
    json_files = glob.glob(os.path.join(source_dir, "**", "*.json"), recursive=True)
        
    print(f"Found {len(json_files)} JSON files in {source_dir} for {dataset_type}.")
    
    if len(json_files) == 0:
        print("No files found. Please check the paths.")
        return

    print("Processing files in parallel... (Running at Maximum Speed)")
    
    # تشغيل المعالجة بأقصى سرعة باستخدام كل أنوية الجهاز
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_file, file_path, output_dir, dataset_type, window_size): file_path for file_path in json_files}
        
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

def main():
    print("Starting Full Dataset Compilation...")
    
    # المسارات زي ما هي موجودة على جهازك
    base_data_dir = r"E:\datasets\snd1156-1-1\data"
    dream_dir = r"E:\datasets\snd1156-1-1\data\dream"
    pinsoro_dir = r"E:\datasets\snd1156-1-1\data\New folder (2)\pinsoro\data"
    
    output_dir = os.path.join(base_data_dir, "processed")
    
    # تشغيل معالجة داتا DREAM فقط
    process_directory(dream_dir, output_dir, dataset_type="DREAM", window_size=300)
    
    # تم إيقاف PInSoRo مؤقتاً لتسريع عملية التدريب على الداتا الأساسية
    # process_directory(pinsoro_dir, output_dir, dataset_type="PInSoRo", window_size=300)
    
    print("Dataset compilation completed.")

if __name__ == "__main__":
    main()