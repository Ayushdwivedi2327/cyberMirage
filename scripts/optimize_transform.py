import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import hashlib
from core.face_pipeline import FacePipeline
from core.crypto_engine import KeyDerivationEngine, CryptoStreamGenerator
from core.metrics_evaluator import PrivacyMetricsEvaluator

pipeline = FacePipeline()

def test_full_engine(img, key, face, strength=1.0):
    h, w = img.shape[:2]
    x, y, fw, fh = [int(v) for v in face['bbox']]
    landmarks = face.get('landmarks', [])

    if landmarks and len(landmarks) >= 5:
        re_x, re_y = landmarks[0]
        le_x, le_y = landmarks[1]
        n_x, n_y = landmarks[2]
        rm_x, rm_y = landmarks[3]
        lm_x, lm_y = landmarks[4]
    else:
        re_x, re_y = x + fw * 0.32, y + fh * 0.38
        le_x, le_y = x + fw * 0.68, y + fh * 0.38
        n_x, n_y = x + fw * 0.50, y + fh * 0.58
        rm_x, rm_y = x + fw * 0.35, y + fh * 0.78
        lm_x, lm_y = x + fw * 0.65, y + fh * 0.78

    eye_dist = max(20.0, float(np.sqrt((le_x - re_x)**2 + (le_y - re_y)**2)))
    weights = CryptoStreamGenerator.generate_keystream_float(
        key, shape=(20,), nonce=hashlib.sha256(key + b'robust_engine_v3').digest()[:16], low=-1.0, high=1.0
    )

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x.copy()
    map_y = grid_y.copy()

    # 1. Non-linear differential eye scaling & tilt
    r_eye = eye_dist * 0.48
    d_re = np.exp(-((grid_x - re_x)**2 + (grid_y - re_y)**2) / (2 * r_eye**2))
    d_le = np.exp(-((grid_x - le_x)**2 + (grid_y - le_y)**2) / (2 * r_eye**2))
    k_eye_w = (0.46 + 0.15 * weights[0]) * strength
    map_x += (- d_re * eye_dist * k_eye_w) + (d_le * eye_dist * k_eye_w)
    map_y += (d_re * eye_dist * 0.18 * weights[1] * strength) - (d_le * eye_dist * 0.18 * weights[2] * strength)

    # 2. Nose apex & columella non-linear displacement
    r_nose = eye_dist * 0.42
    d_nose = np.exp(-((grid_x - n_x)**2 + (grid_y - n_y)**2) / (2 * r_nose**2))
    map_y += d_nose * (eye_dist * (0.50 + 0.15 * weights[3]) * strength)
    map_x += d_nose * (eye_dist * 0.22 * weights[4] * strength)

    # 3. Philtrum & Mouth vertical shift & width
    m_cx = (rm_x + lm_x) * 0.5
    m_cy = (rm_y + lm_y) * 0.5
    r_mouth = eye_dist * 0.50
    d_mouth = np.exp(-((grid_x - m_cx)**2 + (grid_y - m_cy)**2) / (2 * r_mouth**2))
    map_y += - d_mouth * (eye_dist * (0.45 + 0.12 * weights[5]) * strength)

    d_rm = np.exp(-((grid_x - rm_x)**2 + (grid_y - rm_y)**2) / (2 * (eye_dist * 0.32)**2))
    d_lm = np.exp(-((grid_x - lm_x)**2 + (grid_y - lm_y)**2) / (2 * (eye_dist * 0.32)**2))
    k_mw = (0.30 + 0.10 * weights[6]) * strength
    map_x += (- d_rm * eye_dist * k_mw) + (d_lm * eye_dist * k_mw)

    # 4. Jaw & cheek taper
    jaw_cy = y + fh * 0.88
    jaw_cx = x + fw * 0.5
    r_jaw = fw * 0.50
    d_jaw = np.exp(-((grid_x - jaw_cx)**2 + (grid_y - jaw_cy)**2) / (2 * r_jaw**2))
    map_x += (grid_x - jaw_cx) * d_jaw * (0.48 + 0.15 * weights[7]) * strength

    # Smooth Lanczos4 interpolation (continuous, zero noise)
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

    # 5. Keyed Frequency-Domain DCT Orthogonal Rotations
    ycrcb = cv2.cvtColor(warped, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_ch, cr, cb = cv2.split(ycrcb)

    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    y_pad = cv2.copyMakeBorder(y_ch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    cr_pad = cv2.copyMakeBorder(cr, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    cb_pad = cv2.copyMakeBorder(cb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    num_bh = y_pad.shape[0] // 8
    num_bw = y_pad.shape[1] // 8
    angles = CryptoStreamGenerator.generate_keystream_float(
        key, shape=(num_bh * num_bw, 6), nonce=hashlib.sha256(key + b'dct_v3').digest()[:16],
        low=-np.pi * 0.45 * strength, high=np.pi * 0.45 * strength
    )

    b_idx = 0
    for by in range(0, y_pad.shape[0], 8):
        for bx in range(0, y_pad.shape[1], 8):
            if x - 30 <= bx <= x + fw + 30 and y - 30 <= by <= y + fh + 30:
                ang = angles[b_idx]
                d_y = cv2.dct(y_pad[by:by+8, bx:bx+8])
                c1, s1 = np.cos(ang[0]), np.sin(ang[0])
                u1, v1 = d_y[1, 2], d_y[2, 1]
                d_y[1, 2] = c1 * u1 - s1 * v1
                d_y[2, 1] = s1 * u1 + c1 * v1

                c2, s2 = np.cos(ang[1]), np.sin(ang[1])
                u2, v2 = d_y[2, 2], d_y[3, 1]
                d_y[2, 2] = c2 * u2 - s2 * v2
                d_y[3, 1] = s2 * u2 + c2 * v2

                c3, s3 = np.cos(ang[2]), np.sin(ang[2])
                u3, v3 = d_y[1, 3], d_y[3, 2]
                d_y[1, 3] = c3 * u3 - s3 * v3
                d_y[3, 2] = s3 * u3 + c3 * v3
                y_pad[by:by+8, bx:bx+8] = cv2.idct(d_y)

                d_cr = cv2.dct(cr_pad[by:by+8, bx:bx+8])
                c4, s4 = np.cos(ang[3] * 0.6), np.sin(ang[3] * 0.6)
                u4, v4 = d_cr[1, 2], d_cr[2, 1]
                d_cr[1, 2] = c4 * u4 - s4 * v4
                d_cr[2, 1] = s4 * u4 + c4 * v4
                cr_pad[by:by+8, bx:bx+8] = cv2.idct(d_cr)
            b_idx += 1

    y_res = y_pad[:h, :w]
    cr_res = cr_pad[:h, :w]
    cb_res = cb_pad[:h, :w]
    transformed_bgr = cv2.cvtColor(cv2.merge([y_res, cr_res, cb_res]).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (int(x + fw * 0.5), int(y + fh * 0.5)), (int(fw * 0.68), int(fh * 0.74)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 25)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    return (transformed_bgr * mask + img * (1.0 - mask)).astype(np.uint8)

key, _ = KeyDerivationEngine.derive_key('CyberMirage2026!KeyA')

for s in ['sample_sarah.jpg', 'sample_alex.jpg', 'sample_maya.jpg']:
    path = os.path.join('samples', s)
    img = cv2.imread(path)
    face = pipeline.detect_faces(img)[0]
    emb1 = pipeline.extract_embedding(img, face)
    prot = test_full_engine(img, key, face, strength=1.0)
    prot_faces = pipeline.detect_faces(prot)
    emb2 = pipeline.extract_embedding(prot, prot_faces[0] if prot_faces else face)
    sim = pipeline.cosine_similarity(emb1, emb2)
    v = pipeline.match_verdict(emb1, emb2)
    ssim_val = PrivacyMetricsEvaluator.calculate_ssim(img, prot)
    psnr_val = PrivacyMetricsEvaluator.calculate_psnr(img, prot)
    print(f"{s:18s}: CosSim = {sim:.4f} | Defeated = {not v['is_same_identity']} | SSIM = {ssim_val:.4f} | PSNR = {psnr_val:.2f} dB")
