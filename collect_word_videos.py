from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from console_utils import configure_console_encoding, safe_print

configure_console_encoding()

BASE_DIR = Path(__file__).resolve().parent
WORD_DIR = BASE_DIR / 'train_words'
HAND_TASK_PATH = BASE_DIR / 'hand_landmarker.task'
FACE_TASK_PATH = BASE_DIR / 'face_landmarker.task'
RECORD_SECONDS = 4

WORD_LABELS = [
    'สวัสดี', 'ฉัน', 'ชื่อ', 'คุณ', 'อะไร', 'ทำไม', 'ขอโทษ', 'เข้าใจ', 'ไม่', 'ได้',
    'หิว', 'กิน', 'ข้าว', 'ยัง', 'ไม่เป็นไร','กลับบ้าน', 'ป่วย', 'สบายดี', 'นอนหลับ', 'อย่างไร',
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
THAI_FONT_PATHS = [
    r'C:\Windows\Fonts\THSarabun.ttf',
    r'C:\Windows\Fonts\THSarabun New.ttf',
    r'C:\Windows\Fonts\LeelawUI.ttf',
    r'C:\Windows\Fonts\tahoma.ttf',
]


def get_thai_font():
    for path in THAI_FONT_PATHS:
        if Path(path).exists():
            return path
    return None


THAI_FONT = get_thai_font()


def draw_text(frame, text, position, size=28, color=(0, 255, 255)):
    if THAI_FONT is None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(THAI_FONT, size)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def create_landmarkers():
    if not HAND_TASK_PATH.exists() or not FACE_TASK_PATH.exists():
        raise RuntimeError('ต้องมี hand_landmarker.task และ face_landmarker.task ในโฟลเดอร์โปรเจกต์')

    hand_options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=HAND_TASK_PATH.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    face_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=FACE_TASK_PATH.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return (
        mp.tasks.vision.HandLandmarker.create_from_options(hand_options),
        mp.tasks.vision.FaceLandmarker.create_from_options(face_options),
    )


def detect(frame, hand_landmarker, face_landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return hand_landmarker.detect(image), face_landmarker.detect(image)


def draw_landmarks(frame, hand_result, face_result):
    height, width = frame.shape[:2]
    if hand_result and hand_result.hand_landmarks:
        for landmarks in hand_result.hand_landmarks:
            for start, end in HAND_CONNECTIONS:
                first, second = landmarks[start], landmarks[end]
                cv2.line(frame, (int(first.x * width), int(first.y * height)),
                         (int(second.x * width), int(second.y * height)), (0, 255, 0), 2)
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 4, (0, 255, 0), -1)

    if face_result and face_result.face_landmarks:
        for landmarks in face_result.face_landmarks:
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 1, (255, 0, 0), -1)


def record_video(cap, word, hand_landmarker, face_landmarker, window_name):
    output_dir = WORD_DIR / word / 'video'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'clip_{len(list(output_dir.glob("*.mp4")))}.mp4'
    writer = None
    frame_count = 0
    start_time = cv2.getTickCount()

    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < RECORD_SECONDS:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        hand_result, face_result = detect(frame, hand_landmarker, face_landmarker)
        preview = frame.copy()
        draw_landmarks(preview, hand_result, face_result)
        preview = draw_text(preview, f'กำลังบันทึก: {word}', (15, 35), color=(0, 0, 255))
        cv2.imshow(window_name, preview)

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'เปิดไฟล์วิดีโอไม่ได้: {output_path}')
        writer.write(frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if writer is not None:
        writer.release()
    safe_print(f'บันทึก {word}: {frame_count} เฟรม -> {output_path}')


def main():
    try:
        hand_landmarker, face_landmarker = create_landmarkers()
    except Exception as exc:
        safe_print(f'เริ่ม MediaPipe ไม่สำเร็จ: {exc}')
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(0)
    if not cap.isOpened():
        safe_print('ไม่สามารถเปิดกล้องได้')
        return

    window_name = 'Word Video Collection'
    index = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.flip(frame, 1)
            hand_result, face_result = detect(frame, hand_landmarker, face_landmarker)
            preview = frame.copy()
            draw_landmarks(preview, hand_result, face_result)
            preview = draw_text(preview, f'คำ: {WORD_LABELS[index]} ({index + 1}/{len(WORD_LABELS)})', (15, 35))
            preview = draw_text(preview, 'A/D เปลี่ยนคำ | Q อัดวิดีโอ | ESC ออก', (15, 70), size=22, color=(255, 255, 255))
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('a'), ord('A')):
                index = (index - 1) % len(WORD_LABELS)
            elif key in (ord('d'), ord('D')):
                index = (index + 1) % len(WORD_LABELS)
            elif key in (ord('q'), ord('Q')):
                record_video(cap, WORD_LABELS[index], hand_landmarker, face_landmarker, window_name)
            elif key == 27:
                break
    finally:
        cap.release()
        hand_landmarker.close()
        face_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
