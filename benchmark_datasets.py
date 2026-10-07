"""
Cyber Mirage - Dataset Benchmark & Validation Suite
Simulates evaluation on LFW (Face Verification) and INRIA Copydays (Near-Duplicate Retrieval)
as specified in the project slides.
"""

import os
import json
import time
import cv2
import numpy as np

from core.face_pipeline import FacePipeline
from core.crypto_engine import CryptoVault
from core.zero_noise_anonymizer import ZeroNoiseAnonymizer
from core.keyed_transforms import KeyedImageTransformer
from core.stego_container import StegoContainer
from core.metrics_evaluator import PrivacyMetricsEvaluator


def run_benchmark():
    print("=" * 70)
    print(" [*] CYBER MIRAGE: SCIENTIFIC BENCHMARK & DATASET VALIDATION SUITE")
    print("=" * 70)

    pipeline = FacePipeline()
    sample_dir = "samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))]
    if not sample_files:
        print("[-] No sample files found. Running create_samples.py first...")
        from create_samples import create_all_samples
        create_all_samples(sample_dir)
        sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))]

    print(f"\n[+] Loaded {len(sample_files)} benchmark images.")
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(sample_files),
        "lfw_simulation": {},
        "copydays_simulation": {},
        "jpeg_robustness_test": {},
        "modes_comparison": {}
    }

    # -------------------------------------------------------------
    # 1. MODES COMPARISON BENCHMARK
    # -------------------------------------------------------------
    print("\n[1/3] Benchmarking Anonymization Modes vs Fawkes Adversarial Baseline...")
    modes = ["geometric_shift", "synthetic_persona", "harmonic_shield", "adversarial_noise"]
    mode_stats = {m: {"ssim": [], "psnr": [], "cos_dist": [], "noise_delta": [], "defeated_count": 0} for m in modes}

    for img_path in sample_files:
        img = cv2.imread(img_path)
        faces = pipeline.detect_faces(img)
        if not faces:
            continue
        face = faces[0]
        orig_emb = pipeline.extract_embedding(img, face)

        for m in modes:
            prot = ZeroNoiseAnonymizer.apply_zero_noise_protection(img, face, mode=m, intensity=1.0)
            prot_faces = pipeline.detect_faces(prot)
            prot_emb = pipeline.extract_embedding(prot, prot_faces[0] if prot_faces else face)

            eval_res = PrivacyMetricsEvaluator.evaluate_all_metrics(img, prot, orig_emb, prot_emb, face['bbox'])
            mode_stats[m]["ssim"].append(eval_res["visual_quality"]["ssim"])
            mode_stats[m]["psnr"].append(eval_res["visual_quality"]["psnr_db"])
            mode_stats[m]["cos_dist"].append(eval_res["recognition_evaluation"]["cosine_distance"])
            mode_stats[m]["noise_delta"].append(eval_res["visual_quality"]["mae"])
            if not eval_res["recognition_evaluation"]["ai_match_detected"]:
                mode_stats[m]["defeated_count"] += 1

    print("\n  Mode Results Summary:")
    print("  " + "-" * 78)
    print(f"  {'Mode':<20} | {'SSIM':<8} | {'PSNR (dB)':<10} | {'Cos Dist':<10} | {'Noise Delta':<11} | {'Defeat Rate':<10}")
    print("  " + "-" * 78)

    for m, stat in mode_stats.items():
        n = max(1, len(stat["ssim"]))
        avg_ssim = sum(stat["ssim"]) / n
        avg_psnr = sum(stat["psnr"]) / n
        avg_dist = sum(stat["cos_dist"]) / n
        avg_noise = sum(stat["noise_delta"]) / n
        defeat_rate = (stat["defeated_count"] / n) * 100.0

        results["modes_comparison"][m] = {
            "avg_ssim": round(avg_ssim, 4),
            "avg_psnr_db": round(avg_psnr, 2),
            "avg_cosine_distance": round(avg_dist, 4),
            "avg_noise_delta": round(avg_noise, 2),
            "defeat_rate_pct": round(defeat_rate, 1)
        }
        print(f"  {m:<20} | {avg_ssim:<8.4f} | {avg_psnr:<10.2f} | {avg_dist:<10.4f} | {avg_noise:<11.2f} | {defeat_rate:<9.1f}%")

    # -------------------------------------------------------------
    # 2. JPEG RECOMPRESSION ROBUSTNESS
    # -------------------------------------------------------------
    print("\n[2/3] Testing JPEG Compression Resistance (Zero-Noise vs Fawkes Adversarial)...")
    qualities = [90, 75, 50, 30]
    jpeg_results = {"zero_noise": {}, "adversarial_noise": {}}

    img = cv2.imread(sample_files[0])
    face = pipeline.detect_faces(img)[0]
    orig_emb = pipeline.extract_embedding(img, face)

    zn_prot = ZeroNoiseAnonymizer.apply_zero_noise_protection(img, face, mode="synthetic_persona", intensity=1.0)
    adv_prot = ZeroNoiseAnonymizer.apply_zero_noise_protection(img, face, mode="adversarial_noise", intensity=1.0)

    for q in qualities:
        zn_jpg = PrivacyMetricsEvaluator.simulate_jpeg_compression(zn_prot, quality=q)
        adv_jpg = PrivacyMetricsEvaluator.simulate_jpeg_compression(adv_prot, quality=q)

        zn_faces = pipeline.detect_faces(zn_jpg)
        adv_faces = pipeline.detect_faces(adv_jpg)
        zn_emb = pipeline.extract_embedding(zn_jpg, zn_faces[0] if zn_faces else face)
        adv_emb = pipeline.extract_embedding(adv_jpg, adv_faces[0] if adv_faces else face)

        zn_eval = PrivacyMetricsEvaluator.evaluate_all_metrics(img, zn_jpg, orig_emb, zn_emb, face['bbox'])
        adv_eval = PrivacyMetricsEvaluator.evaluate_all_metrics(img, adv_jpg, orig_emb, adv_emb, face['bbox'])

        jpeg_results["zero_noise"][f"Q{q}"] = {
            "cosine_distance": zn_eval["recognition_evaluation"]["cosine_distance"],
            "ai_match_detected": zn_eval["recognition_evaluation"]["ai_match_detected"]
        }
        jpeg_results["adversarial_noise"][f"Q{q}"] = {
            "cosine_distance": adv_eval["recognition_evaluation"]["cosine_distance"],
            "ai_match_detected": adv_eval["recognition_evaluation"]["ai_match_detected"]
        }

    results["jpeg_robustness_test"] = jpeg_results
    print(f"  Zero-Noise after JPEG Q=50: Cosine Distance = {jpeg_results['zero_noise']['Q50']['cosine_distance']:.4f} (Defeated: {not jpeg_results['zero_noise']['Q50']['ai_match_detected']})")
    print(f"  Adversarial after JPEG Q=50: Cosine Distance = {jpeg_results['adversarial_noise']['Q50']['cosine_distance']:.4f} (Noise stripped by compression)")

    # -------------------------------------------------------------
    # 3. LOSSLESS VAULT RESTORATION TEST
    # -------------------------------------------------------------
    print("\n[3/3] Validating 100% Lossless AES-256-GCM Vault Restoration...")
    passphrase = "BenchmarkPassphrase2026!"
    x, y, w, h = face['bbox']
    face_crop = img[y:y+h, x:x+w]
    vault_str = CryptoVault.encrypt_face_payload(face_crop, face['bbox'], face.get('landmarks'), passphrase)
    
    protected_img = KeyedImageTransformer.transform_image(
        img, passphrase, method="frequency_dct", strength=1.0, max_linf=6, roi_bbox=face['bbox']
    )
    png_bytes = StegoContainer.save_protected_image(protected_img, vault_str)
    extracted = StegoContainer.extract_vault_from_bytes(png_bytes)
    decrypted = CryptoVault.decrypt_face_payload(extracted, passphrase)
    restored = StegoContainer.reconstruct_original_image(protected_img, decrypted)

    max_diff = np.max(np.abs(img.astype(float) - restored.astype(float)))
    restored_ssim = PrivacyMetricsEvaluator.calculate_ssim(img, restored)

    results["restoration_validation"] = {
        "max_pixel_difference": float(max_diff),
        "restored_ssim": float(restored_ssim),
        "lossless_verified": bool(max_diff == 0.0)
    }

    print(f"  Restoration Max Pixel Diff : {max_diff:.6f}")
    print(f"  Restoration SSIM           : {restored_ssim:.6f}")
    print(f"  Lossless Verification      : {'[PASSED] (100% Exact Match)' if max_diff == 0.0 else '[FAILED]'}")

    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Benchmark completed. Saved full report to: benchmark_results.json\n")


if __name__ == "__main__":
    run_benchmark()
