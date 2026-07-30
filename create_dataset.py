import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from labels import build_training_label_name, resolve_label_name
from console_utils import configure_console_encoding, safe_print

configure_console_encoding()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
VIDEO_DIR = DATA_DIR / 'video'
OUTPUT_FILE = BASE_DIR / 'data.pickle'
WORD_DATA_DIR = BASE_DIR / 'train_words'
WORD_OUTPUT_FILE = BASE_DIR / 'word_data.pickle'

DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- MediaPipe Tasks API: Hand Landmarker (เหมือนกับ Video.py และ collect_imgs.py) ---
HAND_TASK_PATH = BASE_DIR / 'hand_landmarker.task'
_hand_landmarker = None
if HAND_TASK_PATH.exists():
    try:
        _base_options = mp.tasks.BaseOptions(model_asset_buffer=HAND_TASK_PATH.read_bytes())
        _options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=_base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        _hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(_options)
        safe_print(f'โหลด MediaPipe Hand Landmarker สำเร็จ: {HAND_TASK_PATH}')
    except Exception as e:
        safe_print(f'คำเตือน: โหลด Hand Landmarker ไม่สำเร็จ: {e}')
else:
    safe_print(f'คำเตือน: ไม่พบไฟล์ {HAND_TASK_PATH}')


def extract_features(img):
    """Extract normalized hand landmark features (126 dims: 21 landmarks × 3D × 2 hands)
    using MediaPipe Tasks API เหมือนกับที่ inference_classifier.py ใช้"""
    features = np.zeros(126, dtype=float)

    if _hand_landmarker is None:
        return features

    try:
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = _hand_landmarker.detect(mp_image)

        if result and result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                start_offset = hand_idx * 63
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                wrist = landmarks_array[0]
                middle_tip = landmarks_array[12]
                hand_scale = np.linalg.norm(middle_tip - wrist) + 1e-6
                normalized = (landmarks_array - wrist) / hand_scale
                for i, norm_lm in enumerate(normalized):
                    features[start_offset + (i * 3):(start_offset + (i * 3) + 3)] = norm_lm
    except Exception:
        pass

    return features



def read_image(path):
    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.imread(str(path))
    return image


def process_image_directory(image_dir, label_name):
    for img_path in sorted(image_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        img = read_image(img_path)
        if img is None:
            safe_print(f'Skipping unreadable image: {img_path}')
            continue

        features = extract_features(img)
        data.append(features.tolist())
        labels.append(label_name)


def process_pose_directories(base_dir):
    for class_dir in sorted([path for path in base_dir.iterdir() if path.is_dir() and path.name != 'video']):
        base_label = resolve_label_name(class_dir.name)
        pose_dirs = sorted([path for path in class_dir.iterdir() if path.is_dir()])
        if not pose_dirs:
            process_image_directory(class_dir, base_label)
            continue

        for pose_dir in pose_dirs:
            use_pose_suffix = len(pose_dirs) > 1
            label_name = build_training_label_name(base_label, pose_dir.name if use_pose_suffix else None)
            process_image_directory(pose_dir, label_name)


def process_word_directories(word_base_dir):
    word_data = []
    word_labels = []

    if not word_base_dir.exists():
        return word_data, word_labels

    for word_dir in sorted([path for path in word_base_dir.iterdir() if path.is_dir()]):
        word_label = word_dir.name.strip()
        if not word_label:
            continue

        for file_path in sorted(word_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            img = read_image(file_path)
            if img is None:
                safe_print(f'Skipping unreadable word image: {file_path}')
                continue

            features = extract_features(img)
            word_data.append(features.tolist())
            word_labels.append(word_label)

    return word_data, word_labels


# --- ส่วนที่ 1: อ่านไฟล์รูปภาพปกติจาก data/ (ที่ไม่ใช่โฟลเดอร์ video) ---
data = []
labels = []
process_pose_directories(DATA_DIR)

# --- ส่วนที่ 2: แกะเฟรมจากไฟล์วิดีโอ (.mp4) ใน data/video/ เข้ามาร่วมด้วย ---
if VIDEO_DIR.exists():
    video_class_dirs = sorted([path for path in VIDEO_DIR.iterdir() if path.is_dir()])

    for v_class_dir in video_class_dirs:
        base_label = resolve_label_name(v_class_dir.name)
        pose_dirs = sorted([path for path in v_class_dir.iterdir() if path.is_dir()])
        if not pose_dirs:
            pose_dirs = [v_class_dir]

        for pose_dir in pose_dirs:
            label_name = build_training_label_name(base_label, pose_dir.name)
            for video_path in sorted(pose_dir.iterdir()):
                if not video_path.is_file() or video_path.suffix.lower() not in ['.mp4', '.avi', '.mov']:
                    continue

                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    safe_print(f'Skipping unreadable video: {video_path}')
                    continue

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    features = extract_features(frame)
                    data.append(features.tolist())
                    labels.append(label_name)
                    frame_count += 1

                cap.release()
                safe_print(f'Processed video {video_path.name}: Extracted {frame_count} frames as samples.')

# --- สรุปและบันทึกข้อมูล ---
if not data:
    safe_print('No training data found in data/ or data/video/. Creating an empty dataset file.')
else:
    safe_print(f'Total samples collected from images and videos: {len(data)}')

with OUTPUT_FILE.open('wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

safe_print(f'Saved {len(data)} samples to {OUTPUT_FILE}')

# --- ส่วนที่ 3: สร้างชุดข้อมูลคำจาก train_words/<word>/ ---
word_data, word_labels = process_word_directories(WORD_DATA_DIR)
with WORD_OUTPUT_FILE.open('wb') as f:
    pickle.dump({'data': word_data, 'labels': word_labels}, f)

if word_data:
    safe_print(f'Saved {len(word_data)} word samples to {WORD_OUTPUT_FILE}')
else:
    safe_print(f'No word training samples found in {WORD_DATA_DIR}. Created empty {WORD_OUTPUT_FILE}.')