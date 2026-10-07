"""
Cyber Mirage - Synthetic Sample Face Generator
Generates clean, synthetic portrait faces with diverse characteristics
for benchmarking, testing, and UI demonstration.
"""

import os
import cv2
import numpy as np


def generate_synthetic_portrait(name: str, skin_tone: tuple, eye_color: tuple, hair_color: tuple, width: int = 512, height: int = 512) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Modern studio background gradient
    for y in range(height):
        ratio = y / height
        bg_b = int(220 * (1 - ratio) + 180 * ratio)
        bg_g = int(225 * (1 - ratio) + 190 * ratio)
        bg_r = int(235 * (1 - ratio) + 205 * ratio)
        img[y, :] = (bg_b, bg_g, bg_r)
        
    cx, cy = width // 2, int(height * 0.52)
    face_w, face_h = int(width * 0.23), int(height * 0.31)
    
    # Shoulders / Torso
    cv2.ellipse(img, (cx, height + 40), (int(width * 0.42), int(height * 0.35)), 0, 0, 360, (70, 60, 65), -1)
    # Neck
    cv2.rectangle(img, (cx - int(face_w * 0.45), cy + int(face_h * 0.5)), (cx + int(face_w * 0.45), cy + face_h + 30), skin_tone, -1)
    
    # Hair back
    cv2.ellipse(img, (cx, cy - int(face_h * 0.1)), (face_w + 18, face_h + 20), 0, 0, 360, hair_color, -1)
    
    # Face base
    cv2.ellipse(img, (cx, cy), (face_w, face_h), 0, 0, 360, skin_tone, -1)
    
    # Hair front / top
    cv2.ellipse(img, (cx, cy - int(face_h * 0.65)), (face_w + 10, int(face_h * 0.55)), 0, 180, 360, hair_color, -1)
    
    # Eyes
    eye_dx = int(face_w * 0.48)
    eye_y = cy - int(face_h * 0.15)
    eye_rx, eye_ry = int(face_w * 0.18), int(face_h * 0.10)
    
    for side in (-1, 1):
        ex = cx + side * eye_dx
        # Eye white
        cv2.ellipse(img, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (250, 250, 252), -1)
        cv2.ellipse(img, (ex, eye_y), (eye_rx, eye_ry), 0, 0, 360, (180, 180, 190), 1)
        # Iris
        cv2.circle(img, (ex, eye_y), int(eye_ry * 0.85), eye_color, -1)
        # Pupil
        cv2.circle(img, (ex, eye_y), int(eye_ry * 0.45), (20, 20, 20), -1)
        # Catchlight
        cv2.circle(img, (ex - 2, eye_y - 2), int(eye_ry * 0.22), (255, 255, 255), -1)
        # Eyebrows
        cv2.ellipse(img, (ex, eye_y - eye_ry - 8), (int(eye_rx * 1.15), 5), 0, 180, 360, hair_color, -1)

    # Nose
    nose_y = cy + int(face_h * 0.12)
    nose_color = (max(0, skin_tone[0] - 25), max(0, skin_tone[1] - 25), max(0, skin_tone[2] - 25))
    cv2.line(img, (cx, eye_y + 5), (cx - 4, nose_y), nose_color, 2)
    cv2.line(img, (cx - 4, nose_y), (cx + 6, nose_y + 2), nose_color, 2)
    cv2.circle(img, (cx - 7, nose_y + 1), 3, nose_color, -1)
    cv2.circle(img, (cx + 7, nose_y + 1), 3, nose_color, -1)

    # Mouth
    mouth_y = cy + int(face_h * 0.48)
    lip_color = (max(0, skin_tone[0] - 40), max(0, skin_tone[1] - 50), min(255, skin_tone[2] + 25))
    cv2.ellipse(img, (cx, mouth_y), (int(face_w * 0.38), int(face_h * 0.12)), 0, 0, 180, lip_color, -1)
    cv2.line(img, (cx - int(face_w * 0.38), mouth_y), (cx + int(face_w * 0.38), mouth_y), (80, 70, 140), 2)
    
    # Smooth bilateral polish
    polished = cv2.bilateralFilter(img, d=7, sigmaColor=35, sigmaSpace=35)
    return polished


def create_all_samples(output_dir: str = "samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    samples_config = [
        {
            "name": "sample_alice.png",
            "skin_tone": (195, 210, 245),
            "eye_color": (160, 95, 45),     # Blue
            "hair_color": (35, 45, 80)      # Auburn
        },
        {
            "name": "sample_bob.png",
            "skin_tone": (160, 185, 225),
            "eye_color": (45, 95, 55),      # Green
            "hair_color": (30, 30, 35)      # Dark brown
        },
        {
            "name": "sample_charlie.png",
            "skin_tone": (120, 150, 195),
            "eye_color": (30, 45, 60),      # Deep brown
            "hair_color": (20, 20, 20)      # Black
        }
    ]
    
    for cfg in samples_config:
        filepath = os.path.join(output_dir, cfg["name"])
        img = generate_synthetic_portrait(
            cfg["name"],
            cfg["skin_tone"],
            cfg["eye_color"],
            cfg["hair_color"]
        )
        cv2.imwrite(filepath, img)
        print(f"[+] Created sample portrait: {filepath}")


if __name__ == "__main__":
    create_all_samples()
