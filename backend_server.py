"""HTTP bridge for the Flutter app and the existing pickle models."""
from pathlib import Path
import pickle
from typing import Any
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import mediapipe as mp  # type: ignore[import-not-found]

from labels import resolve_label_name
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Thai Sign Language AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def load_model(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as model_file:
        return pickle.load(model_file)["model"]


character_model = load_model(BASE_DIR / "model.p")
word_model = load_model(BASE_DIR / "word_model.p")
hand_landmarker = None
face_landmarker = None


def create_landmarkers() -> None:
    global hand_landmarker, face_landmarker
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_buffer=(BASE_DIR / "hand_landmarker.task").read_bytes()
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    face_path = BASE_DIR / "face_landmarker.task"
    if face_path.exists():
        face_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=face_path.read_bytes()),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)


def extract_hand_features(image: np.ndarray) -> np.ndarray:
    features = np.zeros(1530, dtype=float)
    if hand_landmarker is None:
        return features
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect(mp_image)
    for hand_index, landmarks in enumerate((result.hand_landmarks or [])[:2]):
        points = np.asarray([[point.x, point.y, point.z] for point in landmarks], dtype=float)
        scale = np.linalg.norm(points[12] - points[0]) + 1e-6
        normalized = (points - points[0]) / scale
        start = hand_index * 63
        features[start:start + 63] = normalized.reshape(-1)
    if face_landmarker is not None:
        face_result = face_landmarker.detect(mp_image)
        if face_result.face_landmarks:
            points = np.asarray([[point.x, point.y, point.z] for point in face_result.face_landmarks[0]], dtype=float)
            normalized = (points - points[1]) / (np.linalg.norm(points[10] - points[1]) + 1e-6)
            features[126:1530] = normalized.reshape(-1)[:1404]
    return features


def detect_presence(image: np.ndarray) -> tuple[np.ndarray, int, bool]:
    features = extract_hand_features(image)
    hand_count = 0
    face_present = False
    if hand_landmarker is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hand_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        hand_count = len(result.hand_landmarks or [])
    if face_landmarker is not None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        face_present = bool(result.face_landmarks)
    return features, hand_count, face_present


def train_word_model_from_storage() -> tuple[Any | None, int]:
    samples: list[np.ndarray] = []
    labels: list[str] = []
    for folder in sorted(path for path in (BASE_DIR / "train_words").iterdir() if path.is_dir()):
        for path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            features, hand_count, _ = detect_presence(image)
            if hand_count:
                samples.append(features)
                labels.append(folder.name)
        video_dir = folder / "video"
        for path in sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []:
            capture = cv2.VideoCapture(str(path))
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                features, hand_count, _ = detect_presence(image)
                if hand_count:
                    samples.append(features)
                    labels.append(folder.name)
            capture.release()
    if len(set(labels)) < 2:
        return None, len(samples)
    trained_model = RandomForestClassifier(random_state=42)
    trained_model.fit(np.asarray(samples), np.asarray(labels))
    with (BASE_DIR / "word_model.p").open("wb") as model_file:
        pickle.dump({"model": trained_model}, model_file)
    return trained_model, len(samples)


if (BASE_DIR / "hand_landmarker.task").exists():
    create_landmarkers()


@app.get("/health")
def health() -> dict[str, bool]:
    return {"character_model": character_model is not None, "word_model": word_model is not None}


@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    raw = await request.body()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    features, hand_count, face_present = detect_presence(image)
    hand_present = hand_count > 0
    character = "-"
    word = "-"
    if character_model is not None and hand_present:
        character = resolve_label_name(str(character_model.predict([features])[0]))
    if word_model is not None and hand_present:
        word = str(word_model.predict([features])[0])
    return {
        "character": character,
        "word": word,
        "hand_present": hand_present,
        "hand_count": hand_count,
        "both_hands": hand_count >= 2,
        "face_present": face_present,
    }


@app.post("/train")
async def train_word(request: Request) -> dict[str, Any]:
    word = request.headers.get("x-word", "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="Missing x-word header")
    image = cv2.imdecode(np.frombuffer(await request.body(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    word_dir = BASE_DIR / "train_words" / word
    word_dir.mkdir(parents=True, exist_ok=True)
    image_path = word_dir / f"mobile_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(image_path), image)

    trained_model, sample_count = train_word_model_from_storage()
    if trained_model is None:
        return {"trained": False, "message": "ต้องมีข้อมูลอย่างน้อย 2 คำ", "samples": sample_count}
    global word_model
    word_model = trained_model
    return {"trained": True, "samples": sample_count, "word": word}


@app.post("/train-video")
async def train_video(request: Request) -> dict[str, Any]:
    word = request.headers.get("x-word", "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="Missing x-word header")
    video_dir = BASE_DIR / "train_words" / word / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"mobile_{uuid.uuid4().hex}.mp4"
    video_path.write_bytes(await request.body())
    trained_model, sample_count = train_word_model_from_storage()
    if trained_model is None:
        return {"trained": False, "message": "ต้องมีข้อมูลวิดีโอ/ภาพอย่างน้อย 2 คำ", "samples": sample_count}
    global word_model
    word_model = trained_model
    return {"trained": True, "samples": sample_count, "word": word, "video": video_path.name}


@app.get("/videos")
def videos() -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for video_path in sorted((BASE_DIR / "train_words").glob("*/video/*")):
        if video_path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        result.append({"word": video_path.parent.parent.name, "name": video_path.name, "url": f"/video/{video_path.parent.parent.name}"})
    return result


@app.get("/video/{word}")
def word_video(word: str) -> FileResponse:
    video_dir = BASE_DIR / "train_words" / word / "video"
    videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}) if video_dir.exists() else []
    if not videos:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(videos[0], media_type="video/mp4", filename=videos[0].name)


app.mount("/", StaticFiles(directory=BASE_DIR / "web_app", html=True), name="web")
