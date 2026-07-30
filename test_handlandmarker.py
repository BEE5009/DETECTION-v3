#!/usr/bin/env python3
"""Test HandLandmarker initialization."""

import mediapipe as mp
from pathlib import Path

# Try to create HandLandmarker
try:
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2
    )
    landmark_detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
    print("HandLandmarker created successfully!")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {e}")
