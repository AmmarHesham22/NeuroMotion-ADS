import os
import glob
import torch
import json
import numpy as np

def build_scaler():
    print("Starting Scaler Generation...")
    
    # 1. تحديد مسار الداتا المُعالجة 
    # (لو بتشغل الكود على Colab، غير المسار ده لـ '/content/processed')
    processed_dir = r"E:\datasets\snd1156-1-1\data\processed" 
    
    pt_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    if not pt_files:
        print(f"Error: No .pt files found in {processed_dir}")
        return
        
    print(f"Found {len(pt_files)} files. Calculating global Mean and Std...")
    
    all_x, all_y, all_c = [], [], []
    
    for f in pt_files:
        data = torch.load(f)
        skel = data["skeleton"] # Shape: [3, T, 17]
        
        # استخراج القيم غير الصفرية فقط (لأن الأصفار بتمثل مفاصل مفقودة ومتبوظش الحسابات)
        # القناة الثالثة skel[2] هي الـ Confidence
        mask = skel[2] > 0.0 
        
        if mask.sum() == 0:
            continue
            
        # فلترة الإحداثيات السليمة فقط
        valid_x = skel[0][mask]
        valid_y = skel[1][mask]
        valid_c = skel[2][mask]
        
        all_x.append(valid_x)
        all_y.append(valid_y)
        all_c.append(valid_c)
        
    # تجميع كل الداتا
    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)
    all_c = torch.cat(all_c)
    
    # 2. حساب المتوسط والانحراف المعياري
    mean_vals = [all_x.mean().item(), all_y.mean().item(), all_c.mean().item()]
    std_vals  = [all_x.std().item(),  all_y.std().item(),  all_c.std().item()]
    
    # 3. تشكيل المصفوفات لتناسب أبعاد الـ Inference اللي هي [3, 1, 1]
    # ده بيسمح بعمل Broadcasting سلس أثناء الـ transform
    mean_array = np.array(mean_vals).reshape(3, 1, 1).tolist()
    std_array = np.array(std_vals).reshape(3, 1, 1).tolist()
    
    scaler_data = {
        "is_fitted": True,
        "mean": mean_array,
        "std": std_array,
        "min_val": None,
        "max_val": None
    }
    
    # 4. حفظ الملف
    # سيتم حفظه في فولدر models
    os.makedirs("models", exist_ok=True)
    out_path = "models/default_scaler.json"
    
    with open(out_path, 'w') as f:
        json.dump(scaler_data, f, indent=4)
        
    print(f"SUCCESS! Scaler saved to {out_path}")
    print(f"Means (X, Y, C): {mean_vals}")
    print(f"Stds (X, Y, C):  {std_vals}")

if __name__ == "__main__":
    build_scaler()