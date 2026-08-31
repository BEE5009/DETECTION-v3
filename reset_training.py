#!/usr/bin/env python3
"""
รีเซ็ตข้อมูลการฝึก: ลบโมเดลเก่าและข้อมูล เตรียมให้เก็บเฉพาะวีดีโอท่าสวัสดี
"""
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.p'
WORD_MODEL_PATH = BASE_DIR / 'word_model.p'
DATA_PATH = BASE_DIR / 'data.pickle'
WORD_DATA_PATH = BASE_DIR / 'word_data.pickle'
TRAIN_WORDS_DIR = BASE_DIR / 'train_words'
HELLO_WORD_DIR = TRAIN_WORDS_DIR / 'สวัสดี'
HELLO_VIDEO_DIR = HELLO_WORD_DIR / 'video'
VIDEO_SOURCE_DIR = BASE_DIR / 'วีดีโอภาษามือ'

print("=" * 60)
print("รีเซ็ตข้อมูลการฝึก")
print("=" * 60)

# 1. ลบไฟล์โมเดลเก่า
for model_file in [MODEL_PATH, WORD_MODEL_PATH]:
    if model_file.exists():
        model_file.unlink()
        print(f"✓ ลบ {model_file.name}")

# 2. ลบไฟล์ข้อมูลการฝึกเก่า
for data_file in [DATA_PATH, WORD_DATA_PATH]:
    if data_file.exists():
        data_file.unlink()
        print(f"✓ ลบ {data_file.name}")

# 3. ลบ train_words/ ทั้งหมด
if TRAIN_WORDS_DIR.exists():
    shutil.rmtree(TRAIN_WORDS_DIR)
    print(f"✓ ลบโฟลเดอร์ {TRAIN_WORDS_DIR}")

# 4. สร้างโฟลเดอร์ใหม่สำหรับสวัสดี
HELLO_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ สร้างโฟลเดอร์ {HELLO_VIDEO_DIR}")

# 5. คัดลอกวีดีโอท่าสวัสดีจาก วีดีโอภาษามือ/
if VIDEO_SOURCE_DIR.exists():
    hello_video_src = VIDEO_SOURCE_DIR / 'สวัสดี.mkv'
    if hello_video_src.exists():
        hello_video_dst = HELLO_VIDEO_DIR / 'สวัสดี.mkv'
        shutil.copy2(hello_video_src, hello_video_dst)
        print(f"✓ คัดลอกวีดีโอท่าสวัสดี: {hello_video_src.name}")
    else:
        print(f"⚠ ไม่พบวีดีโอ: {hello_video_src}")
else:
    print(f"⚠ ไม่พบโฟลเดอร์: {VIDEO_SOURCE_DIR}")

print("=" * 60)
print("✓ รีเซ็ตเสร็จ!")
print("ขั้นตอนถัดไป:")
print("  1. เรียก python create_dataset.py")
print("  2. เรียก python train_classifier.py")
print("  3. เรียก python inference_classifier.py")
print("=" * 60)
