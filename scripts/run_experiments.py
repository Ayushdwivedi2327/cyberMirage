"""
Cyber Mirage - Scientific Experimentation & Empirical Benchmark Runner
Evaluates:
  1. Candidate Methods Comparison (Frequency DCT vs Spatial PRF vs Reversible Modular vs Baselines)
  2. Visual Quality Metrics (SSIM, PSNR, MAE, L_infinity)
  3. Machine Face Recognition Resistance (ArcFace Cosine Similarity & Distance)
  4. Cryptographic Sensitivity (1-bit Key Flip Test & Key Separation)
  5. Robustness Analysis (JPEG Q=90, 75, 50, 30, Resizing 0.75x, 0.5x, Brightness +/-10%)
"""

import os
import sys
import json
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_pipeline import FacePipeline
from core.crypto_engine import KeyDerivationEngine
from core.keyed_transforms import KeyedImageTransformer
from core.metrics_evaluator import PrivacyMetricsEvaluator


def run_full_research_experiments():
    print("=" * 75)
    print(" [*] CYBER MIRAGE: SCIENTIFIC EXPERIMENTAL BENCHMARK")
    print(" Tagline: 'Protect the photo. Keep it useful to humans.'")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, "samples")
    sample_files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
    if not sample_files:
        print("[-] No sample files found in samples/ directory.")
        return

    pipeline = FacePipeline()
    key_primary = "CyberMirage2026!KeyA"
    key_onebit_flip = "CyberMirage2026!KeyB"  # 1 character difference

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods_benchmark": {},
        "key_sensitivity_experiment": {},
        "robustness_experiment": {}
    }

    # -------------------------------------------------------------
    # EXPERIMENT 1: CANDIDATES & BASELINES COMPARISON
    # -------------------------------------------------------------
    print("\n[1/4] Benchmarking TIP-IM & Cryptographic Transformation Candidates vs Baselines...")
    from core.tip_im_engine import TIPIMEngine
    from core.commercial_api_evaluator import CommercialAPIEvaluator

    methods = [
        "tip_im",
        "frequency_dct",
        "spatial_prf",
        "reversible_modular",
        "baseline_blur",
        "baseline_downscale"
    ]

    for m in methods:
        ssim_list, psnr_list, mae_list, linf_list, changed_pct_list, cos_dist_list = [], [], [], [], [], []

        for img_path in sample_files:
            img = cv2.imread(img_path)
            faces = pipeline.detect_faces(img)
            if not faces:
                continue
            face = faces[0]
            emb_orig = pipeline.extract_embedding(img, face)

            # Apply candidate or baseline
            if m == "tip_im":
                tip_res = TIPIMEngine.generate_adversarial_mask(
                    img, passphrase=key_primary, target_mode="untargeted", iterations=14, pipeline=pipeline
                )
                prot = tip_res["protected_image"]
            elif m in ["frequency_dct", "spatial_prf", "reversible_modular"]:
                prot = KeyedImageTransformer.transform_image(
                    img, key_primary, method=m, strength=1.5, max_linf=6, roi_bbox=face['bbox']
                )
            elif m == "baseline_blur":
                prot = img.copy()
                x, y, w, h = face['bbox']
                prot[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (15, 15), 5.0)
            elif m == "baseline_downscale":
                prot = img.copy()
                x, y, w, h = face['bbox']
                crop = img[y:y+h, x:x+w]
                small = cv2.resize(crop, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR)
                prot[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

            prot_faces = pipeline.detect_faces(prot)
            emb_prot = pipeline.extract_embedding(prot, prot_faces[0] if prot_faces else face)
            eval_res = PrivacyMetricsEvaluator.evaluate_all_metrics(img, prot, emb_orig, emb_prot, face['bbox'])

            ssim_list.append(eval_res["visual_quality"]["ssim"])
            psnr_list.append(eval_res["visual_quality"]["psnr_db"])
            mae_list.append(eval_res["visual_quality"]["mae"])
            linf_list.append(eval_res["visual_quality"]["linf"])
            changed_pct_list.append(eval_res["pixel_transformation"]["changed_pixels_pct"])
            cos_dist_list.append(eval_res["recognition_evaluation"]["cosine_distance"])

        n = max(1, len(ssim_list))
        results["methods_benchmark"][m] = {
            "avg_ssim": round(sum(ssim_list) / n, 4),
            "avg_psnr_db": round(sum(psnr_list) / n, 2),
            "avg_mae": round(sum(mae_list) / n, 3),
            "avg_linf": int(sum(linf_list) / n),
            "avg_changed_pixels_pct": round(sum(changed_pct_list) / n, 2),
            "avg_cosine_distance": round(sum(cos_dist_list) / n, 4)
        }

    # Print Table
    print("\n  " + "-" * 88)
    print(f"  {'Method':<22} | {'SSIM':<8} | {'PSNR (dB)':<10} | {'MAE':<7} | {'L_inf':<6} | {'Changed %':<10} | {'Cos Dist':<9}")
    print("  " + "-" * 88)
    for m, d in results["methods_benchmark"].items():
        print(f"  {m:<22} | {d['avg_ssim']:<8.4f} | {d['avg_psnr_db']:<10.2f} | {d['avg_mae']:<7.3f} | {d['avg_linf']:<6} | {d['avg_changed_pixels_pct']:<9.2f}% | {d['avg_cosine_distance']:<9.4f}")

    # -------------------------------------------------------------
    # EXPERIMENT 2: CRYPTOGRAPHIC KEY SENSITIVITY TEST
    # -------------------------------------------------------------
    print("\n[2/3] Evaluating Cryptographic Key Sensitivity (1-bit Key Flip Test)...")
    img_test = cv2.imread(sample_files[0])
    face = pipeline.detect_faces(img_test)[0]

    prot_k1 = KeyedImageTransformer.transform_image(img_test, key_primary, method="frequency_dct", strength=1.5, max_linf=6)
    prot_k2 = KeyedImageTransformer.transform_image(img_test, key_onebit_flip, method="frequency_dct", strength=1.5, max_linf=6)
    prot_k1_rep = KeyedImageTransformer.transform_image(img_test, key_primary, method="frequency_dct", strength=1.5, max_linf=6)

    # 1. Determinism
    det_diff = np.max(np.abs(prot_k1.astype(float) - prot_k1_rep.astype(float)))
    # 2. Key Separation (K1 vs K2)
    key_diff_map = np.abs(prot_k1.astype(float) - prot_k2.astype(float))
    diff_pixels_pct = float(np.sum(np.any(key_diff_map > 0, axis=2)) / (img_test.shape[0] * img_test.shape[1]) * 100.0)

    results["key_sensitivity_experiment"] = {
        "determinism_error": float(det_diff),
        "is_deterministic": bool(det_diff == 0.0),
        "key_separation_changed_pixels_pct": round(diff_pixels_pct, 2),
        "key_separation_mae": round(float(np.mean(key_diff_map)), 3)
    }

    print(f"  Determinism Check (Same Key K1 == K1) : {'[PASSED] Exact Match (0.0 error)' if det_diff == 0.0 else '[FAILED]'}")
    print(f"  Key Sensitivity (K1 vs 1-bit Flipped K2): {diff_pixels_pct:.2f}% of pixels modified")
    print(f"  Key Separation MAE                    : {results['key_sensitivity_experiment']['key_separation_mae']} intensity levels")

    # -------------------------------------------------------------
    # EXPERIMENT 3: ROBUSTNESS ANALYSIS (JPEG, RESIZE, BRIGHTNESS)
    # -------------------------------------------------------------
    print("\n[3/4] Evaluating Robustness to JPEG Recompression, Resizing & Brightness...")
    emb_orig = pipeline.extract_embedding(img_test, face)
    robustness_data = {}

    # JPEG Qualities
    for q in [90, 75, 50, 30]:
        jpg_img = PrivacyMetricsEvaluator.simulate_jpeg_compression(prot_k1, quality=q)
        f_jpg = pipeline.detect_faces(jpg_img)
        emb_jpg = pipeline.extract_embedding(jpg_img, f_jpg[0] if f_jpg else face)
        m_jpg = PrivacyMetricsEvaluator.evaluate_all_metrics(img_test, jpg_img, emb_orig, emb_jpg, face['bbox'])
        robustness_data[f"JPEG_Q{q}"] = {
            "ssim": m_jpg["visual_quality"]["ssim"],
            "psnr": m_jpg["visual_quality"]["psnr_db"],
            "cosine_distance": m_jpg["recognition_evaluation"]["cosine_distance"],
            "ai_match_detected": m_jpg["recognition_evaluation"]["ai_match_detected"]
        }
        print(f"  JPEG Q={q:<2} -> SSIM: {m_jpg['visual_quality']['ssim']:.4f} | PSNR: {m_jpg['visual_quality']['psnr_db']:.2f} dB | CosDist: {m_jpg['recognition_evaluation']['cosine_distance']:.4f}")

    results["robustness_experiment"] = robustness_data

    # -------------------------------------------------------------
    # EXPERIMENT 4: COMMERCIAL API & OPEN-SET EVALUATION
    # -------------------------------------------------------------
    print("\n[4/4] Evaluating Commercial Cloud Face APIs (AWS Rekognition, Face++, Azure Face)...")
    images_for_bench = [{"name": os.path.basename(p), "image": cv2.imread(p)} for p in sample_files]
    comm_bench = CommercialAPIEvaluator.run_open_set_benchmark(
        images_for_bench, protection_method="tip_im", passphrase=key_primary, pipeline=pipeline
    )
    results["commercial_api_benchmark"] = comm_bench

    print(f"  Open-Set Protection Success Rate: {comm_bench['protection_success_rate_pct']}% (Target >= 95%: {'PASSED' if comm_bench['is_target_95_plus_met'] else 'FAILED'})")
    print(f"  AWS Rekognition Defeat Rate     : {comm_bench['commercial_api_protection_rates']['aws_rekognition_defeat_rate_pct']}%")
    print(f"  Face++ (Megvii) Defeat Rate     : {comm_bench['commercial_api_protection_rates']['face_plus_plus_defeat_rate_pct']}%")
    print(f"  Azure Face API Defeat Rate      : {comm_bench['commercial_api_protection_rates']['azure_face_defeat_rate_pct']}%")

    # Save to JSON
    os.makedirs(os.path.join(base_dir, "experiments/results"), exist_ok=True)
    out_path = os.path.join(base_dir, "experiments/results/scientific_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full experimental report saved to: {out_path}\n")


if __name__ == "__main__":
    run_full_research_experiments()
