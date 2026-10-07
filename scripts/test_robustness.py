import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import hashlib
from core.face_pipeline import FacePipeline
from core.crypto_engine import KeyDerivationEngine, CryptoStreamGenerator
from core.metrics_evaluator import PrivacyMetricsEvaluator

def run_test():
    pipeline = FacePipeline()

    def test_multi_scale_flow(img, key, roi_bbox, landmarks, strength=1.0):
        h, w = img.shape[:2]
        x, y, fw, fh = [int(v) for v in roi_bbox]
        
        # Grid coordinates
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x.copy()
        map_y = grid_y.copy()

        # Face center and radius
        cx = x + fw * 0.5
        cy = y + fh * 0.5
        rx = fw * 0.55
        ry = fh * 0.60

        # Elliptical face region envelope with smooth roll-off
        norm_r2 = ((grid_x - cx) / rx)**2 + ((grid_y - cy) / ry)**2
        face_envelope = np.exp(-norm_r2 * 1.8).astype(np.float32)
        face_envelope[norm_r2 > 2.5] = 0.0

        # Generate multi-frequency harmonic flow coefficients deterministically from key
        # 4 octaves of smooth harmonic sine/cosine basis functions
        num_octaves = 5
        seeds = CryptoStreamGenerator.generate_keystream_float(
            key, shape=(num_octaves, 8), nonce=hashlib.sha256(key + b'octave_flow').digest()[:16], low=-1.0, high=1.0
        )

        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)

        # Scale wavelength relative to face size
        base_wavelength = min(fw, fh) * 0.75

        for octave in range(num_octaves):
            freq = (2.0 ** octave) * (2.0 * np.pi / base_wavelength)
            amp = (0.5 ** octave) * (min(fw, fh) * 0.035 * strength)
            
            s = seeds[octave]
            # Keyed angle of wave propagation
            theta_x = s[0] * np.pi
            theta_y = s[1] * np.pi
            phase_x = s[2] * 2.0 * np.pi
            phase_y = s[3] * 2.0 * np.pi
            
            # Keyed smooth directional harmonic deformation
            kx = np.cos(theta_x) * grid_x + np.sin(theta_x) * grid_y
            ky = np.cos(theta_y) * grid_x + np.sin(theta_y) * grid_y
            
            flow_x += amp * np.sin(kx * freq + phase_x)
            flow_y += amp * np.cos(ky * freq + phase_y)

        # Apply smooth flow bounded by face envelope
        map_x += flow_x * face_envelope
        map_y += flow_y * face_envelope

        # Smooth Lanczos4 interpolation (continuous, zero noise)
        warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

        # Apply subtle keyed mid-frequency harmonic tone in LAB space (perceptually imperceptible)
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b_ch = cv2.split(lab)
        
        l_seeds = CryptoStreamGenerator.generate_keystream_float(
            key, shape=(4,), nonce=hashlib.sha256(key + b'lab_harmonic').digest()[:16], low=-1.0, high=1.0
        )
        l_freq = 3.0 * (2.0 * np.pi / base_wavelength)
        l_mod = np.sin((grid_x - cx) * l_freq * l_seeds[0] + (grid_y - cy) * l_freq * l_seeds[1]) * (3.5 * strength) * face_envelope
        l = np.clip(l + l_mod, 0, 255)
        
        result = cv2.cvtColor(cv2.merge([l.astype(np.uint8), a.astype(np.uint8), b_ch.astype(np.uint8)]), cv2.COLOR_LAB2BGR)
        return result

    key, _ = KeyDerivationEngine.derive_key('CyberMirage2026!KeyA')

    for s in ['sample_sarah.jpg', 'sample_alex.jpg', 'sample_maya.jpg']:
        path = os.path.join('samples', s)
        img = cv2.imread(path)
        face = pipeline.detect_faces(img)[0]
        emb1 = pipeline.extract_embedding(img, face)
        
        prot = test_multi_scale_flow(img, key, face['bbox'], face.get('landmarks'), strength=1.0)
        prot_faces = pipeline.detect_faces(prot)
        emb2 = pipeline.extract_embedding(prot, prot_faces[0] if prot_faces else face)
        
        sim = pipeline.cosine_similarity(emb1, emb2)
        v = pipeline.match_verdict(emb1, emb2)
        ssim_val = PrivacyMetricsEvaluator.calculate_ssim(img, prot)
        psnr_val = PrivacyMetricsEvaluator.calculate_psnr(img, prot)
        print(f"{s:18s}: CosSim = {sim:.4f} | Defeated = {not v['is_same_identity']} | SSIM = {ssim_val:.4f} | PSNR = {psnr_val:.2f} dB")

if __name__ == '__main__':
    run_test()

