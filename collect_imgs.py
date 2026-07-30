from pathlib import Path
import sys
from console_utils import configure_console_encoding, safe_print
configure_console_encoding()
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from labels import THAI_LABELS, get_pose_count_for_index

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TASK_PATH = BASE_DIR / 'hand_landmarker.task'

if not MODEL_TASK_PATH.exists():
    safe_print(f"ไม่พบไฟล์ที่: {MODEL_TASK_PATH}")
    safe_print("กรุณาตรวจสอบว่าชื่อไฟล์ถูกต้อง และไม่ได้มีนามสกุลซ่อนอยู่")
    sys.exit(1)

# Initialize body/pose detector (no model file needed)
pose_detector = None
try:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.5
    )
except AttributeError:
    safe_print("Warning: Pose detection not available in this MediaPipe version")

number_of_classes = len(THAI_LABELS)
dataset_size = 100

# --- 1. ตั้งค่า MediaPipe Tasks API ---
base_options = python.BaseOptions(model_asset_buffer=MODEL_TASK_PATH.read_bytes())
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)
# -----------------------------------------------------

# --- 2. ฟังก์ชันสำหรับแสดงผลภาษาไทย ---
def get_thai_font():
    font_paths = [
        r'C:\Windows\Fonts\THSarabun.ttf',
        r'C:\Windows\Fonts\THSarabun New.ttf',
        r'C:\Windows\Fonts\tahoma.ttf',
        r'C:\Windows\Fonts\LeelawUI.ttf'
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return font_path
    return None

THAI_FONT_PATH = get_thai_font()


def draw_thai_text(frame, text, position, font_size=30, color=(255, 255, 0), bg_color=(0, 0, 0)):
    x, y = position
    text_width = max(180, len(text) * font_size // 2)
    text_height = font_size + 8
    cv2.rectangle(frame, (x - 8, y - 6), (x + text_width, y + text_height), bg_color, -1)

    if THAI_FONT_PATH is None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.truetype(THAI_FONT_PATH, font_size)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
# -----------------------------------

def create_camera_capture():
    for index in range(3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
    raise RuntimeError('Unable to open a camera device.')

# --- กำหนดจุดเชื่อมต่อของมือ 21 จุดด้วยตัวเอง (ลด Error จาก mediapipe) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # นิ้วโป้ง
    (0, 5), (5, 6), (6, 7), (7, 8),         # นิ้วชี้
    (5, 9), (9, 10), (10, 11), (11, 12),    # นิ้วกลาง
    (9, 13), (13, 14), (14, 15), (15, 16),  # นิ้วนาง
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # นิ้วก้อย
]

# --- 3. ฟังก์ชันวาดมือที่สร้างขึ้นเอง ---
def draw_landmarks_manually(frame, hand_landmarks):
    h, w, _ = frame.shape

    # วาดเส้นเชื่อมต่อระหว่างข้อต่อมือ
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_lm = hand_landmarks[start_idx]
        end_lm = hand_landmarks[end_idx]

        x1, y1 = int(start_lm.x * w), int(start_lm.y * h)
        x2, y2 = int(end_lm.x * w), int(end_lm.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2) # เส้นสีขาว

    # วาดจุดข้อต่อแต่ละจุด
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) # จุดสีแดง


def draw_task_hand_overlay(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # ตรวจจับมือ
    detection_result = detector.detect(mp_image)

    # ถ้าระบุตำแหน่งมือได้ ให้วาดจุดและเส้น
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            draw_landmarks_manually(frame, hand_landmarks)

    return frame


def write_image(path, image):
    success, buffer = cv2.imencode(path.suffix if path.suffix else '.jpg', image)
    if not success:
        return False
    with path.open('wb') as handle:
        handle.write(buffer)
    return True
# ---------------------------------------------

def main():
    cap = None
    try:
        cap = create_camera_capture()
    except RuntimeError as exc:
        safe_print(exc)
        return

    current_index = 0
    selected_pose = 1
    safe_print('\n--- โหมดเลือกตัวอักษรและท่าทาง ---')
    safe_print('ปุ่ม [A]         : ย้อนกลับคำศัพท์')
    safe_print('ปุ่ม [D]         : ถัดไปคำศัพท์')
    safe_print('ปุ่ม [1]/[2]/[3] : เลือกท่าตามจำนวนที่มี')
    safe_print('ปุ่ม [Q]         : เลือกและเริ่มถ่ายภาพคำศัพท์ปัจจุบัน')
    safe_print('ปุ่ม [ESC]       : ปิดโปรแกรม\n')

    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise RuntimeError('Failed to capture frame from camera.')

                # พลิกภาพซ้าย-ขวาให้เหมือนกระจก
                frame = cv2.flip(frame, 1)

                # วาดโครงกระดูกมือ
                frame = draw_task_hand_overlay(frame)

                # เตรียมข้อความ
                current_label = THAI_LABELS[current_index]
                pose_count = get_pose_count_for_index(current_index)
                if selected_pose > pose_count:
                    selected_pose = 1

                text1 = f'คำศัพท์ปัจจุบัน: {current_label}'
                text2 = f'[{current_index + 1}/{number_of_classes}]'
                text3 = f'ท่า: {selected_pose}/{pose_count}'
                text4 = 'A: ย้อนกลับ | D: ถัดไป | 1/2/3: เลือกท่า | Q: บันทึก | ESC: ออก'

                # วาดภาษาไทย
                frame = draw_thai_text(frame, text1, (20, 10), font_size=32, color=(0, 255, 255))
                frame = draw_thai_text(frame, text2, (20, 50), font_size=24, color=(255, 255, 255))
                frame = draw_thai_text(frame, text3, (20, 85), font_size=24, color=(255, 255, 0))
                frame = draw_thai_text(frame, text4, (20, 120), font_size=18, color=(255, 255, 255))

                cv2.imshow('frame', frame)

                key = cv2.waitKey(25) & 0xFF
                if key == ord('a') or key == ord('A'):
                    current_index = max(0, current_index - 1)
                    selected_pose = 1
                    safe_print(f'>> คำศัพท์ปัจจุบัน: {THAI_LABELS[current_index]}')
                elif key == ord('d') or key == ord('D'):
                    current_index = min(number_of_classes - 1, current_index + 1)
                    selected_pose = 1
                    safe_print(f'>> คำศัพท์ปัจจุบัน: {THAI_LABELS[current_index]}')
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
                    pose_number = int(chr(key))
                    if 1 <= pose_number <= pose_count:
                        selected_pose = pose_number
                        safe_print(f'เลือกท่า {selected_pose}/{pose_count}')
                elif key == ord('q') or key == ord('Q'):
                    safe_print(f'เตรียมบันทึกคำศัพท์: {THAI_LABELS[current_index]} ท่า {selected_pose}')

                    class_dir = DATA_DIR / str(current_index)
                    class_dir.mkdir(parents=True, exist_ok=True)
                    pose_dir = class_dir / str(selected_pose)
                    pose_dir.mkdir(parents=True, exist_ok=True)

                    cv2.waitKey(1000) # หน่วงเวลา 1 วินาทีก่อนถ่ายรูปแรก

                    counter = 0
                    while counter < dataset_size:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.flip(frame, 1)
                        frame = draw_task_hand_overlay(frame)
                        cv2.imshow('frame', frame)
                        cv2.waitKey(25)

                        output_path = pose_dir / f'{counter}.jpg'
                        if write_image(output_path, frame):
                            safe_print(f"บันทึกรูปที่ {counter+1}/{dataset_size} ใน {pose_dir}")
                        else:
                            safe_print(f"ไม่สามารถบันทึกรูปที่ {counter+1}/{dataset_size} ใน {pose_dir}")
                        counter += 1

                    safe_print('บันทึกเสร็จสิ้น! สามารถเลือกคำอื่นต่อได้เลย')

                elif key == 27: # ESC
                    break
            except cv2.error as exc:
                safe_print(f'Unable to display camera feed: {exc}')
                return
    except RuntimeError as exc:
        safe_print(exc)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()