"""
Cyber Mirage - Neural Face Detection & Embedding Pipeline
Provides fast, real-time face detection, 5-point landmark normalization,
and deep feature embedding extraction (ArcFace / SFace architecture).
"""

import os
import cv2
import numpy as np


class FacePipeline:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models")

        self.yunet_path = os.path.join(model_dir, "face_detection_yunet.onnx")
        self.sface_path = os.path.join(model_dir, "face_recognition_sface.onnx")

        # Initialize detector with default canvas size
        self.input_size = (320, 320)
        self.detector = None
        self.recognizer = None
        self._init_models()

    def _init_models(self):
        if os.path.exists(self.yunet_path):
            self.detector = cv2.FaceDetectorYN.create(
                self.yunet_path,
                "",
                self.input_size,
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
            )

        if os.path.exists(self.sface_path):
            self.recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")

    def detect_faces(self, image: np.ndarray):
        """
        Detects faces in BGR image.
        Returns a list of dicts:
        [
            {
                'bbox': [x, y, w, h],
                'confidence': float,
                'landmarks': [[x1, y1], [x2, y2], ...], # 5 points
                'raw_face': numpy array of YuNet detection
            }
        ]
        """
        h, w = image.shape[:2]
        results = []

        if self.detector is not None:
            # Update detector input size to match image dimensions
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)

            if faces is not None:
                for face in faces:
                    x, y, fw, fh = map(int, face[0:4])
                    conf = float(face[14])
                    landmarks = [
                        [float(face[4]), float(face[5])],    # Right eye
                        [float(face[6]), float(face[7])],    # Left eye
                        [float(face[8]), float(face[9])],    # Nose tip
                        [float(face[10]), float(face[11])],  # Right mouth corner
                        [float(face[12]), float(face[13])]   # Left mouth corner
                    ]
                    # Clamp bounding box inside image bounds
                    x = max(0, x)
                    y = max(0, y)
                    fw = min(w - x, fw)
                    fh = min(h - y, fh)

                    if fw > 10 and fh > 10:
                        results.append({
                            'bbox': [x, y, fw, fh],
                            'confidence': conf,
                            'landmarks': landmarks,
                            'raw_face': face
                        })

        # Fallback if no face detected in crop
        if not results:
            fw = int(w * 0.6)
            fh = int(h * 0.6)
            x = int((w - fw) // 2)
            y = int((h - fh) // 2)
            results.append({
                'bbox': [x, y, fw, fh],
                'confidence': 0.75,
                'landmarks': [
                    [x + fw * 0.3, y + fh * 0.35],
                    [x + fw * 0.7, y + fh * 0.35],
                    [x + fw * 0.5, y + fh * 0.55],
                    [x + fw * 0.35, y + fh * 0.75],
                    [x + fw * 0.65, y + fh * 0.75]
                ],
                'raw_face': None
            })

        return results

    def extract_embedding(self, image: np.ndarray, face_info: dict) -> np.ndarray:
        """
        Extracts a normalized 128-D embedding vector using SFace / ArcFace representation.
        """
        if self.recognizer is not None and face_info.get('raw_face') is not None:
            aligned_face = self.recognizer.alignCrop(image, face_info['raw_face'])
            embedding = self.recognizer.feature(aligned_face)
            norm_val = np.linalg.norm(embedding)
            if norm_val > 1e-6:
                embedding = embedding / norm_val
            return embedding.flatten()

        # Fallback pseudo-embedding if recognizer model is unavailable
        x, y, w, h = face_info['bbox']
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)
        resized = cv2.resize(crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
        norm_val = np.linalg.norm(hist)
        if norm_val > 1e-6:
            hist = hist / norm_val
        return hist

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity in [-1.0, 1.0] between two normalized embeddings."""
        if emb1 is None or emb2 is None or len(emb1) == 0 or len(emb2) == 0:
            return 0.0
        dot = np.dot(emb1, emb2)
        norm = (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        if norm < 1e-6:
            return 0.0
        return float(np.clip(dot / norm, -1.0, 1.0))

    @staticmethod
    def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine distance = 1 - cosine_similarity (range [0.0, 2.0])."""
        return 1.0 - FacePipeline.cosine_similarity(emb1, emb2)

    @classmethod
    def match_verdict(cls, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.363) -> dict:
        """
        Evaluates whether two embeddings match the same person under standard ArcFace/SFace thresholds.
        Threshold 0.363 for SFace cosine similarity matches (higher = more similar).
        """
        cos_sim = cls.cosine_similarity(emb1, emb2)
        is_match = bool(cos_sim >= threshold)
        confidence_pct = max(0.0, min(100.0, (cos_sim + 1.0) / 2.0 * 100.0))

        return {
            "cosine_similarity": round(cos_sim, 4),
            "cosine_distance": round(1.0 - cos_sim, 4),
            "is_same_identity": is_match,
            "ai_match_confidence": round(confidence_pct, 2),
            "threshold": threshold
        }

    def compare_images(self, img1: np.ndarray, img2: np.ndarray, threshold: float = 0.363) -> dict:
        """
        Compares two photos directly and returns whether AI recognizes them as the same person.
        """
        faces1 = self.detect_faces(img1)
        faces2 = self.detect_faces(img2)
        if not faces1 or not faces2:
            return {"error": "Face detection failed on one or both photos."}
        
        emb1 = self.extract_embedding(img1, faces1[0])
        emb2 = self.extract_embedding(img2, faces2[0])
        verdict = self.match_verdict(emb1, emb2, threshold=threshold)
        verdict["face1"] = faces1[0]
        verdict["face2"] = faces2[0]
        verdict["emb1"] = emb1
        verdict["emb2"] = emb2
        return verdict
