# Sign Language Writer Mobile

Flutter Android app for the existing Thai sign-language models.

## Architecture

- Flutter captures a front-camera frame and displays the result UI.
- `backend_server.py` loads `model.p` and `word_model.p`, extracts the same hand features as the desktop app, and returns JSON.
- `/video/<word>` streams the first training video under `train_words/<word>/video/`.
- `/videos` lists every synchronized training video.
- `/train` adds one camera frame to a word and retrains the shared word model.
- `/train-video` uploads a mobile recording, stores it under the same word, and retrains the shared word model.
- `/predict` returns the character, word, hand count (including two-hand detection), and face status.

## Run the backend

From the repository root:

```powershell
pip install fastapi uvicorn
python -m uvicorn backend_server:app --host 0.0.0.0 --port 8000
```

## Create and run Android project

Flutter SDK is required. From this folder:

```powershell
flutter create . --platforms android
flutter pub get
flutter run
```

For a physical Android phone, replace `10.0.2.2` in `lib/main.dart` with the computer's LAN IP, for example:

```powershell
flutter run --dart-define=BACKEND_URL=http://192.168.1.20:8000
```

The Android emulator uses `10.0.2.2` to reach the host computer. Add camera and Internet permissions to `android/app/src/main/AndroidManifest.xml` if Flutter does not add them automatically:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

The existing model is a Python/scikit-learn pickle, so it stays on the backend. For a fully offline app later, export a compatible model to TensorFlow Lite or ONNX and replace the HTTP call.
