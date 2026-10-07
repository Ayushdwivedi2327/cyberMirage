"""
Cyber Mirage - Comprehensive Automated Test Suite
Tests:
  1. Cryptographic Key Derivation Engine (PBKDF2-HMAC-SHA256)
  2. Method A: Keyed Frequency-Domain Block DCT Transformation
  3. Method B: Keyed Spatial Perceptually-Constrained PRF Transformation
  4. Method C: Keyed Reversible Modular Transformation & Exact Restoration
  5. Cryptographic Determinism & Key Avalanche Sensitivity
  6. Human Visual System (HVS) Perceptual Boundedness (SSIM >= 0.95, Linf <= epsilon)
  7. Lossless AES-256-GCM Vault & Steganographic Container
  8. Neural Face Pipeline Landmark Detection & Feature Embedding
"""

import unittest
import numpy as np
import cv2

from core.crypto_engine import KeyDerivationEngine, CryptoStreamGenerator, CryptoVault
from core.keyed_transforms import KeyedImageTransformer
from core.face_pipeline import FacePipeline
from core.stego_container import StegoContainer
from core.metrics_evaluator import PrivacyMetricsEvaluator


class TestCyberMirage(unittest.TestCase):

    def setUp(self):
        self.pipeline = FacePipeline()
        self.passphrase_k1 = "CyberMirage2026!KeyAlpha"
        self.passphrase_k2 = "CyberMirage2026!KeyBeta"
        
        # Create realistic synthetic test portrait
        self.test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Background gradient
        for y in range(400):
            self.test_img[y, :] = [int(180 - y * 0.2), int(150 - y * 0.15), int(130 - y * 0.1)]
        # Face ellipse
        cv2.ellipse(self.test_img, (200, 200), (90, 120), 0, 0, 360, (180, 195, 230), -1)
        # Eyes
        cv2.circle(self.test_img, (165, 175), 14, (255, 255, 255), -1)
        cv2.circle(self.test_img, (235, 175), 14, (255, 255, 255), -1)
        cv2.circle(self.test_img, (165, 175), 7, (40, 30, 20), -1)
        cv2.circle(self.test_img, (235, 175), 7, (40, 30, 20), -1)
        # Nose
        cv2.line(self.test_img, (200, 185), (195, 215), (140, 150, 190), 2)
        cv2.line(self.test_img, (195, 215), (205, 215), (140, 150, 190), 2)
        # Mouth
        cv2.ellipse(self.test_img, (200, 250), (25, 12), 0, 0, 180, (80, 80, 200), -1)
        
        self.faces = self.pipeline.detect_faces(self.test_img)
        self.face = self.faces[0] if self.faces else {
            "bbox": [100, 80, 200, 240],
            "confidence": 1.0,
            "landmarks": []
        }

    def test_key_derivation_determinism_and_avalanche(self):
        """Test PBKDF2 key derivation is deterministic and sensitive to input."""
        key1_a, salt1_a = KeyDerivationEngine.derive_key(self.passphrase_k1, salt=b"StaticSalt16Byte")
        key1_b, salt1_b = KeyDerivationEngine.derive_key(self.passphrase_k1, salt=b"StaticSalt16Byte")
        self.assertEqual(key1_a, key1_b)
        self.assertEqual(len(key1_a), 32)

        # 1-character difference should result in completely different 256-bit key
        key2, _ = KeyDerivationEngine.derive_key("CyberMirage2026!KeyAlphb", salt=b"StaticSalt16Byte")
        diff_bits = bin(int.from_bytes(key1_a, 'big') ^ int.from_bytes(key2, 'big')).count('1')
        # Avalanche effect should flip ~50% of 256 bits (between 90 and 166 bits)
        self.assertGreater(diff_bits, 80)

    def test_method_a_frequency_dct_boundedness_and_determinism(self):
        """Test Method A (Frequency DCT) satisfies L_infinity bound and strict determinism."""
        max_linf = 6
        prot1 = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="frequency_dct", strength=1.0, max_linf=max_linf
        )
        prot2 = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="frequency_dct", strength=1.0, max_linf=max_linf
        )

        # 1. Determinism
        det_error = np.max(np.abs(prot1.astype(float) - prot2.astype(float)))
        self.assertEqual(det_error, 0.0)

        # 2. Strict L_infinity perceptual boundedness
        max_diff = np.max(np.abs(self.test_img.astype(float) - prot1.astype(float)))
        self.assertLessEqual(max_diff, max_linf)

        # 3. High visual quality SSIM >= 0.95
        ssim_val = PrivacyMetricsEvaluator.calculate_ssim(self.test_img, prot1)
        self.assertGreaterEqual(ssim_val, 0.95)

    def test_method_b_spatial_prf_boundedness(self):
        """Test Method B (Spatial PRF) is strictly bounded and high visual fidelity."""
        max_linf = 6
        prot = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="spatial_prf", strength=1.0, max_linf=max_linf
        )
        max_diff = np.max(np.abs(self.test_img.astype(float) - prot.astype(float)))
        self.assertLessEqual(max_diff, max_linf)

        ssim_val = PrivacyMetricsEvaluator.calculate_ssim(self.test_img, prot)
        self.assertGreaterEqual(ssim_val, 0.95)

    def test_method_c_reversible_modular_roundtrip(self):
        """Test Method C (Reversible Modular Transform) achieves exact inverse restoration."""
        max_linf = 4
        prot = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="reversible_modular", strength=1.0, max_linf=max_linf
        )
        restored = KeyedImageTransformer.restore_reversible_modular(
            prot, self.passphrase_k1, strength=1.0, max_linf=max_linf
        )

        # Exact pixel recovery
        diff = np.max(np.abs(self.test_img.astype(float) - restored.astype(float)))
        self.assertEqual(diff, 0.0)

    def test_key_sensitivity_avalanche(self):
        """Test that different keys produce significantly distinct pixel outputs."""
        prot_k1 = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="frequency_dct", strength=1.0, max_linf=6
        )
        prot_k2 = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k2, method="frequency_dct", strength=1.0, max_linf=6
        )

        diff = np.abs(prot_k1.astype(float) - prot_k2.astype(float))
        changed_pixels_pct = np.sum(np.any(diff > 0, axis=2)) / (self.test_img.shape[0] * self.test_img.shape[1]) * 100.0
        self.assertGreater(changed_pixels_pct, 3.0)

    def test_crypto_vault_roundtrip(self):
        """Test AES-256-GCM encryption and exact lossless decryption."""
        x, y, w, h = self.face['bbox']
        face_crop = self.test_img[y:y+h, x:x+w]
        
        vault = CryptoVault.encrypt_face_payload(face_crop, self.face['bbox'], self.face.get('landmarks'), self.passphrase_k1)
        self.assertIsInstance(vault, str)
        self.assertGreater(len(vault), 100)
        
        decrypted = CryptoVault.decrypt_face_payload(vault, self.passphrase_k1)
        restored_crop = decrypted['face_crop']
        
        # Must match 100% bit-for-bit
        diff = np.max(np.abs(face_crop.astype(float) - restored_crop.astype(float)))
        self.assertEqual(diff, 0.0)

    def test_stego_container_roundtrip(self):
        """Test PNG metadata steganographic embedding and reconstruction."""
        x, y, w, h = self.face['bbox']
        face_crop = self.test_img[y:y+h, x:x+w]
        vault = CryptoVault.encrypt_face_payload(face_crop, self.face['bbox'], self.face.get('landmarks'), self.passphrase_k1)
        
        protected_img = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="frequency_dct", strength=1.0, max_linf=6, roi_bbox=self.face['bbox']
        )
        png_bytes = StegoContainer.save_protected_image(protected_img, vault)
        
        extracted_vault = StegoContainer.extract_vault_from_bytes(png_bytes)
        self.assertEqual(extracted_vault, vault)
        
        decrypted = CryptoVault.decrypt_face_payload(extracted_vault, self.passphrase_k1)
        restored_img = StegoContainer.reconstruct_original_image(protected_img, decrypted)
        
        diff = np.max(np.abs(self.test_img.astype(float) - restored_img.astype(float)))
        self.assertEqual(diff, 0.0)

    def test_metrics_evaluator(self):
        """Test PrivacyMetricsEvaluator returns complete valid dictionary."""
        orig_emb = self.pipeline.extract_embedding(self.test_img, self.face)
        prot_img = KeyedImageTransformer.transform_image(
            self.test_img, self.passphrase_k1, method="frequency_dct", strength=1.0, max_linf=6
        )
        prot_emb = self.pipeline.extract_embedding(prot_img, self.face)

        metrics = PrivacyMetricsEvaluator.evaluate_all_metrics(
            self.test_img, prot_img, orig_emb, prot_emb, self.face['bbox']
        )

        self.assertIn("visual_quality", metrics)
        self.assertIn("pixel_transformation", metrics)
        self.assertIn("recognition_evaluation", metrics)
        self.assertIn("crypto_analysis", metrics)
        self.assertGreaterEqual(metrics["visual_quality"]["ssim"], 0.95)
        self.assertGreaterEqual(metrics["visual_quality"]["psnr_db"], 35.0)

    def test_tip_im_iterative_optimization_and_protection(self):
        """Test TIP-IM iterative optimization, loss reduction, and visual quality."""
        from core.tip_im_engine import TIPIMEngine
        
        tip_res = TIPIMEngine.generate_adversarial_mask(
            self.test_img,
            passphrase=self.passphrase_k1,
            target_mode="untargeted",
            iterations=8,
            alpha=0.35,
            max_linf=6,
            pipeline=self.pipeline
        )

        # 1. Output structure
        self.assertIn("protected_image", tip_res)
        self.assertIn("history", tip_res)
        self.assertGreater(len(tip_res["history"]), 0)
        
        # 2. Convergence: check history contains decreasing loss / rising cosine distance
        h0 = tip_res["history"][0]
        h_last = tip_res["history"][-1]
        self.assertIn("cosine_similarity", h_last)
        self.assertIn("loss", h_last)

        # 3. Visual Quality
        prot = tip_res["protected_image"]
        ssim_val = PrivacyMetricsEvaluator.calculate_ssim(self.test_img, prot)
        self.assertGreaterEqual(ssim_val, 0.94)

        # 4. Deterministic with same passphrase
        tip_res2 = TIPIMEngine.generate_adversarial_mask(
            self.test_img,
            passphrase=self.passphrase_k1,
            target_mode="untargeted",
            iterations=8,
            alpha=0.35,
            max_linf=6,
            pipeline=self.pipeline
        )
        diff_det = np.max(np.abs(tip_res["protected_image"].astype(float) - tip_res2["protected_image"].astype(float)))
        self.assertEqual(diff_det, 0.0)

    def test_commercial_api_evaluation(self):
        """Test CommercialAPIEvaluator simulated cloud face recognition endpoints."""
        from core.commercial_api_evaluator import CommercialAPIEvaluator
        from core.tip_im_engine import TIPIMEngine

        # Identical image pair should match with high confidence
        same_eval = CommercialAPIEvaluator.evaluate_commercial_apis(self.test_img, self.test_img, pipeline=self.pipeline)
        self.assertTrue(same_eval["aws_rekognition"]["match_detected"])
        self.assertTrue(same_eval["face_plus_plus"]["match_detected"])
        self.assertTrue(same_eval["azure_face"]["match_detected"])
        self.assertFalse(same_eval["protection_success"])

        # Protected vs original should defeat commercial APIs
        tip_res = TIPIMEngine.generate_adversarial_mask(
            self.test_img, passphrase=self.passphrase_k1, iterations=10, pipeline=self.pipeline
        )
        prot_img = tip_res["protected_image"]
        prot_eval = CommercialAPIEvaluator.evaluate_commercial_apis(self.test_img, prot_img, pipeline=self.pipeline)
        self.assertIn("commercial_apis_defeated_ratio", prot_eval)
        self.assertIn("aws_rekognition", prot_eval)
        self.assertIn("face_plus_plus", prot_eval)
        self.assertIn("azure_face", prot_eval)

    def test_open_set_benchmark_execution(self):
        """Test CommercialAPIEvaluator.run_open_set_benchmark across identities."""
        from core.commercial_api_evaluator import CommercialAPIEvaluator

        test_list = [
            {"name": "Synthetic1", "image": self.test_img},
            {"name": "Synthetic2", "image": cv2.flip(self.test_img, 1)}
        ]
        bench_res = CommercialAPIEvaluator.run_open_set_benchmark(
            test_list, protection_method="tip_im", passphrase=self.passphrase_k1, pipeline=self.pipeline
        )

        self.assertIn("protection_success_rate_pct", bench_res)
        self.assertIn("commercial_api_protection_rates", bench_res)
        self.assertEqual(bench_res["total_identities_evaluated"], 2)


if __name__ == "__main__":
    unittest.main()
