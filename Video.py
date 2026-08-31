import os
import pickle
import sys
import time
from pathlib import Path
from console_utils import configure_console_encoding, safe_print
configure_console_encoding()

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from labels import THAI_LABELS
except ImportError:
    safe_print("ไม่พบไฟล์ labels.py จะใช้ Label เริ่มต้นแทน")
    THAI_LABELS = ['ก', 'ข', 'ค']

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data' / 'video'
MODEL_PATH = BASE_DIR / 'video_model.p'
DATA_DIR.mkdir(parents=True, exist_ok=True)

RECORD_SECONDS = 6

THAI_FONT_PATHS = [
    r'C:\Windows\Fonts\THSarabun.ttf',
    r'C:\Windows\Fonts\THSarabun New.ttf',
    r'C:\Windows\Fonts\LeelawUI.ttf',
    r'C:\Windows\Fonts\LeelaUIb.ttf',
    r'C:\Windows\Fonts\tahoma.ttf',
]

MY_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def get_thai_font():
    for font_path in THAI_FONT_PATHS:
        if Path(font_path).exists():
            return font_path
    return None

THAI_FONT = get_thai_font()

def draw_thai_text(frame, text, position, size=28, color=(0, 255, 0)):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)

    if THAI_FONT is not None:
        font = ImageFont.truetype(THAI_FONT, size)
    else:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def create_camera_capture():
    for index in range(3):
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
            
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60)
            return cap
    raise RuntimeError('ไม่สามารถเปิดกล้องได้')

def create_hand_landmarker():
    model_path = BASE_DIR / 'hand_landmarker.task'
    if not model_path.exists():
        safe_print(f"คำเตือน: ไม่พบไฟล์โมเดลข้อต่อมือที่ {model_path}")
        return None

    base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,                          
        min_hand_detection_confidence=0.8,    
        min_hand_presence_confidence=0.8      
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)

HAND_LANDMARKER = None

def get_hand_landmarker():
    global HAND_LANDMARKER
    if HAND_LANDMARKER is None:
        HAND_LANDMARKER = create_hand_landmarker()
    return HAND_LANDMARKER

def draw_simple_hand_overlay(frame):
    if not hasattr(frame, 'shape'):
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    landmarker = get_hand_landmarker()
    if landmarker is None: return frame
        
    try:
        result = landmarker.detect(image)
    except Exception: return frame

    if result is None or not result.hand_landmarks: return frame

    for hand_landmarks in result.hand_landmarks:
        for landmark in hand_landmarks:
            cx = int(landmark.x * frame.shape[1])
            cy = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1) 

        for connection in MY_HAND_CONNECTIONS:
            start = hand_landmarks[connection[0]]
            end = hand_landmarks[connection[1]]
            start_pt = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_pt = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            cv2.line(frame, start_pt, end_pt, (255, 0, 0), 2) 

    return frame

def extract_features_from_frame_with_landmarks(frame):
    features = np.zeros(126, dtype=float)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    landmarker = get_hand_landmarker()
    if landmarker is None: return features
         
    try:
        result = landmarker.detect(image)
    except Exception: return features

    if not result or not result.hand_landmarks: return features

    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
        start_offset = hand_idx * 63
        for lm_idx, landmark in enumerate(hand_landmarks):
            features[start_offset + (lm_idx * 3)] = landmark.x
            features[start_offset + (lm_idx * 3) + 1] = landmark.y
            features[start_offset + (lm_idx * 3) + 2] = landmark.z

    return features

def extract_features_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: break
        frames.append(frame)
    cap.release()

    if not frames: return None

    sample_indices = np.linspace(0, len(frames) - 1, num=min(8, len(frames)), dtype=int)
    features = []
    for index in sample_indices:
        features.append(extract_features_from_frame_with_landmarks(frames[index]))

    return np.mean(np.stack(features, axis=0), axis=0)

def collect_videos():
    safe_print("กำลังเตรียมเปิดกล้อง...")
    try:
        cap = create_camera_capture()
    except RuntimeError as exc:
        safe_print(exc); time.sleep(3); return

    current_index = 0
    safe_print('\n--- เข้าสู่โหมดเก็บข้อมูลวิดีโอ 60 FPS ---')
    safe_print('ปุ่ม [A] / [D]   : เปลี่ยนคำศัพท์ภาษาไทย')
    safe_print('ปุ่ม [Q]         : เริ่มบันทึกวิดีโอ (ความยาวครั้งละ 2 วินาที ที่ความเร็ว 60 FPS)')
    safe_print('ปุ่ม [ESC]       : ปิดโปรแกรม\n')

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            frame = cv2.flip(frame, 1) 
            display_frame = draw_simple_hand_overlay(frame.copy())
            label_name = THAI_LABELS[current_index]
            
            display_frame = draw_thai_text(display_frame, f'คำศัพท์ปัจจุบัน: {label_name}', (15, 30), size=26, color=(0, 255, 0))
            display_frame = draw_thai_text(display_frame, 'กด A: ย้อนกลับ | D: ถัดไป | Q: เริ่มบันทึก | ESC: ออก', (15, 65), size=18, color=(255, 255, 0))
            
            cv2.imshow('Video Collection (60 FPS Version)', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a'):
                current_index = (current_index - 1) % len(THAI_LABELS)
                safe_print(f'>> เปลี่ยนคำศัพท์เป็น: {THAI_LABELS[current_index]}')
            elif key == ord('d'):
                current_index = (current_index + 1) % len(THAI_LABELS)
                safe_print(f'>> เปลี่ยนคำศัพท์เป็น: {THAI_LABELS[current_index]}')
            elif key == 27: break
            elif key == ord('q'):
                class_dir = DATA_DIR / str(current_index)
                class_dir.mkdir(parents=True, exist_ok=True)
                clip_count = len(list(class_dir.glob('*.mp4')))
                output_path = class_dir / f'clip_{clip_count}.mp4'

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, 60.0, (frame.shape[1], frame.shape[0]))

                safe_print(f"กำลังบันทึกวิดีโอแบบ 60 FPS: {label_name}... เริ่มขยับมือได้เลยครับ")
                start_time = cv2.getTickCount()
                while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < RECORD_SECONDS:
                    ret, frame = cap.read()
                    if not ret or frame is None: break
                    frame = cv2.flip(frame, 1)
                    display_frame = draw_simple_hand_overlay(frame.copy())
                    display_frame = draw_thai_text(display_frame, f'● กำลังบันทึกท่าทาง (60 FPS): {label_name}', (15, 30), size=26, color=(0, 0, 255))
                    cv2.imshow('Video Collection (60 FPS Version)', display_frame)
                    writer.write(frame)
                    cv2.waitKey(1)

                writer.release()
                safe_print(f'บันทึกวิดีโอ 60 FPS สำเร็จ: {output_path}')
    finally:
        cap.release(); cv2.destroyAllWindows()

def train_video_model():
    safe_print("กำลังเริ่มฝึกสอนสมอง AI ด้วยชุดข้อมูลพิกัดมือ...")
    X, y = [], []
    class_dirs = sorted([path for path in DATA_DIR.iterdir() if path.is_dir()])
    
    for class_dir in class_dirs:
        label_index = class_dir.name
        if not label_index.isdigit(): continue
        for video_path in sorted(class_dir.glob('*.mp4')):
            features = extract_features_from_video(video_path)
            if features is None or np.all(features == 0): continue
            X.append(features)
            y.append(int(label_index))

    if not X:
        safe_print('Error: ไม่พบชุดข้อมูลวิดีโอในระบบ กรุณารันโหมดอัดวิดีโอก่อนครับ (`python Video.py --collect`)')
        return False

    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(np.unique(y)) < 2:
        safe_print('Error: จำเป็นต้องมีข้อมูลอย่างน้อย 2 คำศัพท์ขึ้นไปในการเทรนโมเดลครับ')
        return False

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(x_train, y_train)

    safe_print(f'ความแม่นยำของการทดสอบ (Test Accuracy Score): {accuracy_score(model.predict(x_test), y_test) * 100:.2f}%')
    with MODEL_PATH.open('wb') as f:
        pickle.dump({'model': model, 'labels': THAI_LABELS}, f)
    safe_print(f'ฝึกสอนสมอง AI สำเร็จ! บันทึกไฟล์เรียบร้อยที่: {MODEL_PATH}')
    return True

def live_inference():
    safe_print("กำลังเปิดระบบตรวจจับท่าทางเรียลไทม์...")
    if not MODEL_PATH.exists():
        safe_print(f"Error: ไม่พบไฟล์สมอง AI ที่ {MODEL_PATH} กรุณารันโหมดเทรนก่อนครับ (`python Video.py --train`)" ); return

    with MODEL_PATH.open('rb') as f:
        data = pickle.load(f)
        model, labels = data['model'], data['labels']
    labels_dict = {index: label for index, label in enumerate(labels)}

    cap = create_camera_capture()
    frame_buffer = [] 

    safe_print('\n--- เข้าสู่โหมดใช้งานจริงเรียลไทม์ 60 FPS (Live Mode) ---')
    safe_print('ทำท่าทางภาษามือหน้ากล้องได้เลย (ภาพลื่นไหลสูงสุด) | กด [ESC] เพื่อปิดโปรแกรม\n')

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: break

            frame = cv2.flip(frame, 1)
            display_frame = draw_simple_hand_overlay(frame.copy())
            
            frame_buffer.append(frame)
            # >>> ขยายบัฟเฟอร์เป็น 60 เฟรม เพื่อให้สอดคล้องกับความเร็ว 60 เฟรมต่อวินาที
            if len(frame_buffer) > 60: frame_buffer.pop(0) 

            predicted_text = "กำลังรอสัญญาณมือ..."
            if len(frame_buffer) >= 30: # รอเฟรมสะสมครึ่งวินาที (30 เฟรม) เพื่อเริ่มวิเคราะห์
                sample_indices = np.linspace(0, len(frame_buffer) - 1, num=min(8, len(frame_buffer)), dtype=int)
                features = [extract_features_from_frame_with_landmarks(frame_buffer[idx]) for idx in sample_indices]
                feature_vector = np.mean(np.stack(features, axis=0), axis=0)
                
                if np.all(feature_vector == 0):
                    predicted_text = "ไม่พบมือในรัศมีกล้อง"
                else:
                    pred = model.predict([feature_vector])
                    predicted_text = labels_dict.get(int(pred[0]), str(pred[0]))

            display_frame = draw_thai_text(display_frame, f'คำทำนายภาษามือ: {predicted_text}', (15, 35), size=30, color=(0, 255, 0))
            cv2.imshow('Live Sign Language Motion Detector (60 FPS)', display_frame)
            if cv2.waitKey(1) & 0xFF == 27: break 
    finally:
        cap.release(); cv2.destroyAllWindows()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true', help='เปิดระบบอัดวิดีโอท่าทางภาษามือ')
    parser.add_argument('--train', action='store_true', help='สั่งเทรนสมอง AI จากวิดีโอที่เก็บไว้')
    parser.add_argument('--live', action='store_true', help='เปิดกล้องใช้งานตรวจจับจริงแบบเรียลไทม์')
    args = parser.parse_args()

    if args.train:
        if train_video_model():
            safe_print('การเทรนเสร็จสมบูรณ์ กำลังเข้าสู่โหมดใช้งานจริง...')
            live_inference()
    elif args.live:
        live_inference()
    else:
        collect_videos()

if __name__ == '__main__':
    main()