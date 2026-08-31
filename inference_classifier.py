import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence, cast
import cv2
from console_utils import configure_console_encoding, safe_print
configure_console_encoding()
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import threading
import io
import tkinter as tk
from tkinter import simpledialog
from pygame import mixer
from sklearn.ensemble import RandomForestClassifier
from labels import THAI_LABELS, get_pose_count_for_label, resolve_label_name


def draw_hand_landmarks_on_frame(frame: np.ndarray, hand_landmarks: Sequence[Any]) -> np.ndarray:
    h, w, _ = frame.shape
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]
    for connection in hand_connections:
        start_idx, end_idx = connection
        start_lm = hand_landmarks[start_idx]
        end_lm = hand_landmarks[end_idx]
        x1, y1 = int(start_lm.x * w), int(start_lm.y * h)
        x2, y2 = int(end_lm.x * w), int(end_lm.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    
    return frame


def draw_face_landmarks_on_frame(frame: np.ndarray, face_landmarks: Sequence[Any]) -> np.ndarray:
    h, w, _ = frame.shape
    for landmark in face_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    return frame


def draw_body_landmarks_on_frame(frame: np.ndarray, pose_landmarks: Sequence[Any]) -> np.ndarray:
    h, w, _ = frame.shape
    body_connections = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (23, 24), (23, 25), (24, 26),
        (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32)
    ]
    
    for connection in body_connections:
        start_idx, end_idx = connection
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            start_lm = pose_landmarks[start_idx]
            end_lm = pose_landmarks[end_idx]
            if start_lm.visibility > 0.5 and end_lm.visibility > 0.5:
                x1, y1 = int(start_lm.x * w), int(start_lm.y * h)
                x2, y2 = int(end_lm.x * w), int(end_lm.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    for landmark in pose_landmarks:
        if landmark.visibility > 0.5:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)
    
    return frame

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False

USE_BLINK_TRIGGER = False
VALID_THAI_LABELS = {label for label in THAI_LABELS}
for label in THAI_LABELS:
    pose_count = get_pose_count_for_label(label)
    for pose in range(1, pose_count + 1):
        if pose_count > 1:
            VALID_THAI_LABELS.add(f'{label}-ท่า{pose}')

try:
    stdout_reconfigure = getattr(sys.stdout, 'reconfigure', None)
    if callable(stdout_reconfigure):
        stdout_reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.p'
WORD_DATA_DIR = BASE_DIR / 'train_words'
WORD_MODEL_PATH = BASE_DIR / 'word_model.p'
WORD_DATA_DIR.mkdir(parents=True, exist_ok=True)

# เริ่มต้นระบบเสียง (เรียกใช้งานครั้งเดียว)
mixer.init()

# ตั้งค่าฟอนต์ภาษาไทยสำหรับแสดงบนหน้าจอ
THAI_FONT_PATHS = [
    r'C:\Windows\Fonts\THSarabun.ttf',
    r'C:\Windows\Fonts\THSarabun New.ttf',
    r'C:\Windows\Fonts\LeelawUI.ttf',
    r'C:\Windows\Fonts\tahoma.ttf',
]

def get_thai_font() -> str | None:
    for font_path in THAI_FONT_PATHS:
        if Path(font_path).exists(): return font_path
    return None

THAI_FONT = get_thai_font()

def draw_thai_text(frame: np.ndarray, text: str, position: tuple[int, int], size: int = 32, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(THAI_FONT, size) if THAI_FONT else ImageFont.load_default()

    x, y = position
    outline_color = (20, 20, 20)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def fit_frame_to_box(frame: np.ndarray, box_width: int, box_height: int) -> np.ndarray:
    panel = np.full((box_height, box_width, 3), (255, 247, 240), dtype=np.uint8)
    if frame is None or box_width <= 0 or box_height <= 0:
        return panel

    src_h, src_w = frame.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return panel

    scale = min(box_width / src_w, box_height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    offset_x = (box_width - new_w) // 2
    offset_y = (box_height - new_h) // 2
    panel[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return panel


def build_layout_frame(
    camera_frame: np.ndarray,
    current_word: str,
    predicted_character: str,
    predicted_word: str,
    hand_present: bool,
    blink_enabled: bool,
    character_enabled: bool,
    word_enabled: bool,
) -> np.ndarray:
    frame_h, frame_w = camera_frame.shape[:2]
    canvas = np.full((frame_h, frame_w, 3), (255, 246, 238), dtype=np.uint8)

    right_panel_w = max(190, int(frame_w * 0.28))
    left_panel_w = frame_w - right_panel_w
    top_panel_h = max(220, int(frame_h * 0.67))
    bottom_panel_h = frame_h - top_panel_h

    x_split = left_panel_w
    y_split = top_panel_h

    border_color = (220, 140, 90)
    cv2.rectangle(canvas, (0, 0), (frame_w - 1, frame_h - 1), border_color, 2)
    cv2.line(canvas, (x_split, 0), (x_split, frame_h - 1), border_color, 2)
    cv2.line(canvas, (0, y_split), (left_panel_w - 1, y_split), border_color, 2)

    camera_box = fit_frame_to_box(camera_frame, left_panel_w - 10, top_panel_h - 10)
    canvas[5:5 + camera_box.shape[0], 5:5 + camera_box.shape[1]] = camera_box

    canvas = draw_thai_text(canvas, "ช่อง A", (20, y_split + 10), size=26, color=(255, 255, 255))
    a_text = current_word if current_word else "(ยังไม่มีตัวอักษร)"
    canvas = draw_thai_text(canvas, a_text, (20, y_split + 55), size=40, color=(255, 255, 255))
    word_text = f"คำที่ตรวจพบ: {predicted_word if word_enabled and predicted_word else 'รอคำ'}"
    canvas = draw_thai_text(canvas, word_text, (20, y_split + 105), size=26, color=(255, 255, 255))

    sidebar_x = x_split + 16
    canvas = draw_thai_text(canvas, "ฟังก์ชัน", (sidebar_x, 16), size=30, color=(255, 255, 255))

    live_text = f"ท่าปัจจุบัน: {predicted_character if predicted_character else 'รอท่าทาง'}"
    hand_text = "สองมือ: พร้อม" if hand_present else "สองมือ: รอ"
    blink_text = "Blink: ON" if blink_enabled else "Blink: OFF"
    mode_text = f"พยัญชนะ: {'ON' if character_enabled else 'OFF'} | คำ: {'ON' if word_enabled else 'OFF'}"

    canvas = draw_thai_text(canvas, live_text, (sidebar_x, 68), size=20, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, hand_text, (sidebar_x, 98), size=20, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, blink_text, (sidebar_x, 128), size=20, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, mode_text, (sidebar_x, 155), size=16, color=(255, 255, 255))

    canvas = draw_thai_text(canvas, "C: เปิด/ปิดตรวจพยัญชนะ", (sidebar_x, 180), size=18, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, "V: เปิด/ปิดตรวจคำ", (sidebar_x, 206), size=18, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, "R: ให้บอทพูดผลลัพธ์", (sidebar_x, 232), size=18, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, "T: ลบตัวล่าสุด", (sidebar_x, 258), size=18, color=(255, 255, 255))
    canvas = draw_thai_text(canvas, "Q: ออกจากโปรแกรม", (sidebar_x, 284), size=18, color=(255, 255, 255))

    return canvas


def get_layout_regions(frame_shape: tuple[int, int, int]) -> dict[str, tuple[int, int, int, int]]:
    frame_h, frame_w = frame_shape[:2]
    right_panel_w = max(190, int(frame_w * 0.28))
    left_panel_w = frame_w - right_panel_w
    top_panel_h = max(220, int(frame_h * 0.67))
    bottom_panel_h = frame_h - top_panel_h
    return {
        'a_box': (0, top_panel_h, left_panel_w, bottom_panel_h),
    }


def is_point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < (rx + rw) and ry <= y < (ry + rh)


def prompt_text_for_a_box(current_text: str) -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        value = simpledialog.askstring(
            'แก้ไขช่อง A',
            'พิมพ์ข้อความที่ต้องการแสดงในช่อง A',
            initialvalue=current_text,
            parent=root,
        )
    finally:
        root.destroy()

    if value is None:
        return None
    return value.strip()


def on_mouse_click(event: int, x: int, y: int, _flags: int, state: Any | None) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if not isinstance(state, dict):
        return

    frame_shape = state.get('last_frame_shape')
    if frame_shape is None:
        return

    a_box = get_layout_regions(frame_shape)['a_box']
    if is_point_in_rect(x, y, a_box):
        state['request_text_input'] = True

# --- Initialize Face & Body Detection ---
CV2_DATA_DIR = Path(cv2.__file__).resolve().parent / 'data'
FACE_CASCADE = cv2.CascadeClassifier(str(CV2_DATA_DIR / 'haarcascade_frontalface_default.xml'))
EYE_CASCADE = cv2.CascadeClassifier(str(CV2_DATA_DIR / 'haarcascade_eye.xml'))

pose_detector = None
try:
    mp_solutions = getattr(mp, 'solutions', None)
    if mp_solutions is not None:
        mp_pose = cast(Any, mp_solutions).pose
        pose_detector = cast(Any, mp_pose).Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
except AttributeError:
    pass


def create_hand_landmarker() -> Any | None:
    try:
        model_path = BASE_DIR / 'hand_landmarker.task'
        if not model_path.exists():
            safe_print(f"Warning: ไม่พบไฟล์ hand_landmarker.task ที่ {model_path}")
            return None
        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)
    except Exception as e:
        safe_print(f"Warning: Failed to initialize hand detector: {e}")
        return None


def create_face_landmarker() -> Any | None:
    try:
        model_path = BASE_DIR / 'face_landmarker.task'
        if not model_path.exists():
            return None
        base_options = mp.tasks.BaseOptions(model_asset_buffer=model_path.read_bytes())
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        return None


HAND_LANDMARKER: Any | None = create_hand_landmarker()
FACE_LANDMARKER: Any | None = create_face_landmarker()

def extract_image_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))
    return np.asarray(gray, dtype=float).reshape(-1) / 255.0


def extract_hand_features(frame: np.ndarray) -> np.ndarray:
    features = np.zeros(1530, dtype=float)

    hand_landmarker = HAND_LANDMARKER
    if hand_landmarker is None:
        return features

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = hand_landmarker.detect(mp_image)

        if result and result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                start_offset = hand_idx * 63
                landmarks_array = cast(np.ndarray, np.asarray([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=float))
                wrist = landmarks_array[0]
                middle_tip = landmarks_array[12]
                hand_scale = np.linalg.norm(middle_tip - wrist) + 1e-6
                normalized = (landmarks_array - wrist) / hand_scale
                for i, norm_lm in enumerate(normalized):
                    features[start_offset + (i * 3):(start_offset + (i * 3) + 3)] = norm_lm

            if FACE_LANDMARKER is not None:
                face_result = FACE_LANDMARKER.detect(mp_image)
                if face_result and face_result.face_landmarks:
                    face_array = np.asarray(
                        [[lm.x, lm.y, lm.z] for lm in face_result.face_landmarks[0]],
                        dtype=float,
                    )
                    face_origin = face_array[1]
                    face_scale = np.linalg.norm(face_array[10] - face_origin) + 1e-6
                    face_features = (face_array - face_origin) / face_scale
                    features[126:126 + min(face_features.size, 1404)] = face_features.reshape(-1)[:1404]
    except Exception:
        pass

    return features


def is_hand_present(frame: np.ndarray) -> bool:
    hand_landmarker = HAND_LANDMARKER
    if hand_landmarker is None:
        return False

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = hand_landmarker.detect(mp_image)
        return bool(result and len(result.hand_landmarks) >= 1)
    except Exception:
        return False


def detect_eye_state(frame: np.ndarray) -> tuple[bool, list[tuple[int, int]], bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) == 0:
        return False, [], False

    x, y, w, h = faces[0]
    roi_gray = cast(Any, gray)[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(cast(Any, roi_gray), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    eye_points: list[tuple[int, int]] = []
    for (ex, ey, ew, eh) in eyes[:2]:
        eye_points.append((x + ex + ew // 2, y + ey + eh // 2))

    if len(eye_points) >= 2:
        return False, eye_points, True


    face_center = (x + w // 2, y + h // 2)
    return True, [face_center], True


def save_word_training_sample(word_label: str, frame: np.ndarray) -> Path | None:
    normalized_word = word_label.strip()
    if not normalized_word:
        return None

    word_dir = WORD_DATA_DIR / normalized_word
    word_dir.mkdir(parents=True, exist_ok=True)
    sample_index = len(list(word_dir.glob('*.jpg')))
    output_path = word_dir / f'{sample_index}.jpg'

    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return None

    with output_path.open('wb') as f:
        f.write(buffer)

    return output_path


def train_word_model_from_folders() -> bool:
    samples = []
    labels = []

    for word_dir in sorted([path for path in WORD_DATA_DIR.iterdir() if path.is_dir()]):
        word_label = word_dir.name.strip()
        if not word_label:
            continue

        for image_path in sorted(word_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue

            raw = np.fromfile(str(image_path), dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is None:
                continue

            features = extract_hand_features(img)
            samples.append(features)
            labels.append(word_label)

    if not samples:
        return False

    model = RandomForestClassifier(random_state=42)
    model.fit(np.asarray(samples, dtype=float), np.asarray(labels))

    with WORD_MODEL_PATH.open('wb') as f:
        pickle.dump({'model': model}, f)

    return True


def speak_word_worker(text_to_speak: str) -> None:
    if not text_to_speak.strip():
        return
    if not GTTS_AVAILABLE:
        return
    try:
        tts_factory = gTTS
        if tts_factory is None:
            return
        tts = tts_factory(text=text_to_speak, lang='th')
        fp = io.BytesIO()
        tts.write_to_fp(cast(Any, fp))
        fp.seek(0)
        
        mixer.music.load(fp, 'mp3')
        mixer.music.play()
    except Exception:
        pass

def main() -> None:
    global USE_BLINK_TRIGGER

    if not MODEL_PATH.exists() and not WORD_MODEL_PATH.exists():
        safe_print("ไม่พบไฟล์ model.p หรือ word_model.p กรุณารัน train_classifier.py ก่อน")
        return

    model = None
    word_model = None
    if MODEL_PATH.exists():
        with MODEL_PATH.open('rb') as f:
            model = pickle.load(f)['model']
    if WORD_MODEL_PATH.exists():
        with WORD_MODEL_PATH.open('rb') as f:
            word_model = pickle.load(f)['model']

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(0)
    
    current_word = ""
    predicted_character = ""
    predicted_word = ""
    character_enabled = model is not None
    word_enabled = word_model is not None
    word_candidate = ""
    word_started_at = 0.0
    spoken_word = ""
    eyes_were_closed = False
    


    window_name = 'Sign Language Writer'
    ui_state: dict[str, Any] = {
        'request_text_input': False,
        'last_frame_shape': None,
    }
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click, ui_state)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        hand_present = is_hand_present(frame)
        predicted_character = ""
        predicted_word = ""
        _hand_result = None
        if hand_present:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                hand_landmarker = HAND_LANDMARKER
                if hand_landmarker is not None:
                    _hand_result = cast(Any, hand_landmarker).detect(mp_image)
            except Exception:
                pass
            features = extract_hand_features(frame)
            if character_enabled and model is not None:
                pred = model.predict([features])
                raw_prediction = str(pred[0])
                predicted_character = resolve_label_name(raw_prediction)
                if predicted_character not in VALID_THAI_LABELS:
                    predicted_character = ""
            if word_enabled and word_model is not None:
                predicted_word = str(word_model.predict([features])[0])

        if word_enabled and predicted_word and hand_present:
            if predicted_word != spoken_word:
                if predicted_word != word_candidate:
                    word_candidate = predicted_word
                    word_started_at = time.monotonic()
                elif time.monotonic() - word_started_at >= 1.0:
                    spoken_word = predicted_word
                    threading.Thread(
                        target=speak_word_worker,
                        args=(predicted_word,),
                        daemon=True,
                    ).start()
                    safe_print(f"ตรวจพบคำ '{predicted_word}' ครบ 1 วินาที กำลังพูด")
        else:
            word_candidate = ''
            word_started_at = 0.0
            spoken_word = ''

        # Draw hand landmarks on frame (reuse result from detection above)
        try:
            if _hand_result and _hand_result.hand_landmarks:
                for hand_landmarks in _hand_result.hand_landmarks:
                    frame = draw_hand_landmarks_on_frame(frame, hand_landmarks)
        except Exception:
            pass
        
        try:
            if FACE_LANDMARKER:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                face_result = FACE_LANDMARKER.detect(mp_image)
                if face_result and face_result.face_landmarks:
                    for face_landmarks in face_result.face_landmarks:
                        frame = draw_face_landmarks_on_frame(frame, face_landmarks)
        except Exception:
            pass
        
        pose_detector_local = cast(Any, pose_detector) if pose_detector is not None else None
        if pose_detector_local is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose_detector_local.process(frame_rgb)
                if pose_result and pose_result.pose_landmarks:
                    frame = draw_body_landmarks_on_frame(frame, cast(Sequence[Any], pose_result.pose_landmarks.landmark))
            except Exception:
                pass

        # ตรวจจับตาและวาดจุดสถานะ: ลืมตา=น้ำเงิน, หลับตา=แดง
        is_closed, eye_points, face_found = detect_eye_state(frame)
        point_color = (0, 0, 255) if is_closed else (255, 0, 0)
        for point in eye_points:
            cv2.circle(frame, point, 8, point_color, -1)

        if not hand_present or not face_found:
            eyes_were_closed = False
        elif is_closed and not eyes_were_closed and predicted_character:
            current_word += predicted_character
            safe_print(f"บันทึกอักษร: {predicted_character} -> คำปัจจุบันคือ: {current_word}")
            eyes_were_closed = True
        elif not is_closed:
            eyes_were_closed = False

        display_frame = build_layout_frame(
            camera_frame=frame,
            current_word=current_word,
            predicted_character=predicted_character,
            predicted_word=predicted_word,
            hand_present=hand_present,
            blink_enabled=is_closed,
            character_enabled=character_enabled,
            word_enabled=word_enabled,
        )
        ui_state['last_frame_shape'] = display_frame.shape
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):
            if model is not None:
                character_enabled = not character_enabled
                safe_print(f"ตรวจพยัญชนะ: {'เปิด' if character_enabled else 'ปิด'}")
        elif key == ord('v') or key == ord('V'):
            if word_model is not None:
                word_enabled = not word_enabled
                safe_print(f"ตรวจคำ: {'เปิด' if word_enabled else 'ปิด'}")
        elif key == ord('r') or key == ord('R'):
            speech_text = predicted_word if word_enabled and predicted_word else current_word
            safe_print(f"กำลังส่งผลลัพธ์ '{speech_text}' ไปให้ AI ประมวลผลเสียง...")
            threading.Thread(target=speak_word_worker, args=(speech_text,), daemon=True).start()
        elif key == ord('t') or key == ord('T'):
            if current_word:
                current_word = current_word[:-1]
                safe_print(f"ลบตัวอักษรล่าสุดแล้ว: {current_word}")
            else:
                safe_print("ไม่มีตัวอักษรให้ลบ")
        elif key == ord('e') or key == ord('E'):
            if character_enabled and predicted_character:
                current_word += predicted_character
                safe_print(f"บันทึกอักษรด้วยปุ่ม E: {predicted_character} -> คำปัจจุบันคือ: {current_word}")
            else:
                safe_print("ยังไม่มีพยัญชนะให้บันทึก")
        elif key == ord('y') or key == ord('Y'):
            safe_print("ระบบจับท่ามือพร้อมใช้งาน")
        elif key == ord('w') or key == ord('W'):
            if not current_word.strip():
                safe_print('ยังไม่มีคำสำหรับเทรน')
                continue

            saved_path = save_word_training_sample(current_word, frame)
            if saved_path is None:
                safe_print('บันทึกตัวอย่างคำไม่สำเร็จ')
                continue

            trained = train_word_model_from_folders()
            if trained:
                safe_print(f"เทรนคำสำเร็จ: {current_word} | บันทึกตัวอย่างที่ {saved_path}")
                safe_print(f"บันทึกโมเดลคำแล้วที่: {WORD_MODEL_PATH}")
            else:
                safe_print('ยังไม่มีข้อมูลพอสำหรับเทรนคำ')
        elif key == ord('q') or key == ord('Q'):
            break

        if ui_state.get('request_text_input'):
            ui_state['request_text_input'] = False
            typed_text = prompt_text_for_a_box(current_word)
            if typed_text is not None:
                current_word = typed_text
                safe_print(f"อัปเดตช่อง A เป็น: {current_word}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()