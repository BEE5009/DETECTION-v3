#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test all main files for runtime errors"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

print("=" * 60)
print("Testing imports...")
print("=" * 60)

# Test 1: create_dataset
try:
    from create_dataset import extract_features, _hand_landmarker
    print("✓ create_dataset.py imported")
    print(f"  - Hand Landmarker: {'Loaded' if _hand_landmarker else 'Not loaded'}")
except Exception as e:
    print(f"✗ create_dataset.py failed: {e}")
    sys.exit(1)

# Test 2: train_classifier
try:
    from train_classifier import load_training_data
    print("✓ train_classifier.py imported")
    data = load_training_data()
    print(f"  - Training samples: {len(data['data'])}")
except Exception as e:
    print(f"✗ train_classifier.py failed: {e}")
    sys.exit(1)

# Test 3: inference_classifier
try:
    from inference_classifier import HAND_LANDMARKER, FACE_LANDMARKER, extract_hand_features
    print("✓ inference_classifier.py imported")
    print(f"  - Hand Landmarker: {'Loaded' if HAND_LANDMARKER else 'Not loaded'}")
    print(f"  - Face Landmarker: {'Loaded' if FACE_LANDMARKER else 'Not loaded'}")
except Exception as e:
    print(f"✗ inference_classifier.py failed: {e}")
    sys.exit(1)

# Test 4: Video
try:
    from Video import create_hand_landmarker, get_hand_landmarker
    print("✓ Video.py imported")
    landmarker = get_hand_landmarker()
    print(f"  - Hand Landmarker: {'Loaded' if landmarker else 'Not loaded'}")
except Exception as e:
    print(f"✗ Video.py failed: {e}")
    sys.exit(1)

# Test 5: collect_imgs
try:
    from collect_imgs import detector, draw_task_hand_overlay
    print("✓ collect_imgs.py imported")
    print(f"  - Hand Landmarker detector: {'Available' if detector else 'Not available'}")
except Exception as e:
    print(f"✗ collect_imgs.py failed: {e}")
    sys.exit(1)

print("=" * 60)
print("All imports successful!")
print("=" * 60)
