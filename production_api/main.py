import os
import sys

# ==========================================
# الحل الجذري لمشكلة الـ Imports في المشروع بالكامل
# ==========================================
# 1. تحديد مسار فولدر production_api
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. تحديد مسار المشروع الرئيسي (NeuroMotion-ADS)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
# 3. تحديد مسار فولدر neuromotion_core
CORE_DIR = os.path.join(ROOT_DIR, "neuromotion_core")

# إضافة كل المسارات دي لـ Python مرة واحدة عشان يشوف كل الملفات
if CURRENT_DIR not in sys.path: sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)
if CORE_DIR not in sys.path: sys.path.insert(0, CORE_DIR)
# ==========================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# دلوقتي الاستدعاء ده هيشتغل صاروخ
from routers import inference

app = FastAPI(
    title="NeuroMotion-ADS AI Service",
    description="Python inference backend for the NeuroMotion-ADS foundation model.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "NeuroMotion-ADS AI Service"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)