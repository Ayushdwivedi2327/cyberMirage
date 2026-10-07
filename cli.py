"""
Cyber Mirage - Command Line Interface (CLI)
Provides batch protection and decryption commands for images and directories.

Usage:
  python cli.py protect --input input.jpg --output protected.png --password MySecret123 --mode tip_im
  python cli.py decrypt --input protected.png --output restored.png --password MySecret123
  python cli.py benchmark
"""

import os
import sys
import argparse
import cv2
import numpy as np

from core.face_pipeline import FacePipeline
from core.crypto_engine import CryptoVault
from core.keyed_transforms import KeyedImageTransformer
from core.tip_im_engine import TIPIMEngine
from core.commercial_api_evaluator import CommercialAPIEvaluator
from core.stego_container import StegoContainer
from core.metrics_evaluator import PrivacyMetricsEvaluator


def protect_command(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        print(f"Error: Could not read image: {args.input}")
        sys.exit(1)

    pipeline = FacePipeline()
    faces = pipeline.detect_faces(image_bgr)
    if not faces:
        print("Warning: No faces detected in input image.")
        sys.exit(1)

    face = faces[0]
    x, y, w, h = face['bbox']
    orig_emb = pipeline.extract_embedding(image_bgr, face)

    print(f"[*] Detected face at [{x}, {y}, {w}, {h}] (Confidence: {face['confidence']:.2f})")
    print(f"[*] Encrypting biometric face vault with AES-256-GCM...")
    vault_str = CryptoVault.encrypt_face_payload(
        image_bgr[y:y+h, x:x+w], face['bbox'], face.get('landmarks'), args.password
    )

    if args.mode == "tip_im":
        print(f"[*] Executing TIP-IM (Targeted Identity-Protection Iterative Method, {args.iterations} iterations)...")
        tip_res = TIPIMEngine.generate_adversarial_mask(
            image_bgr,
            passphrase=args.password,
            target_mode=args.target_mode,
            iterations=args.iterations,
            max_linf=args.max_linf,
            pipeline=pipeline
        )
        protected_bgr = tip_res["protected_image"]
        metrics = tip_res["metrics"]
        cos_dist = tip_res["final_cosine_dist"]
        is_defeated = tip_res["is_defeated"]
    else:
        print(f"[*] Applying Keyed Cryptographic Transformation (Method: {args.mode})...")
        protected_bgr = KeyedImageTransformer.transform_image(
            image_bgr,
            args.password,
            method=args.mode,
            strength=args.strength,
            max_linf=args.max_linf,
            roi_bbox=face['bbox'],
            landmarks=face.get('landmarks')
        )
        prot_faces = pipeline.detect_faces(protected_bgr)
        prot_emb = pipeline.extract_embedding(protected_bgr, prot_faces[0] if prot_faces else face)
        metrics = PrivacyMetricsEvaluator.evaluate_all_metrics(
            image_bgr, protected_bgr, orig_emb, prot_emb, face['bbox']
        )
        cos_dist = metrics["recognition_evaluation"]["cosine_distance"]
        is_defeated = not metrics["recognition_evaluation"]["ai_match_detected"]

    print("\n--- Privacy & Resistance Evaluation ---")
    print(f"  Protection Status        : {'MATCH DEFEATED (PROTECTED)' if is_defeated else 'MATCH NOT FULLY BROKEN'}")
    print(f"  ArcFace Cosine Distance  : {cos_dist:.4f} (Threshold >= 0.637)")
    print(f"  Visual Quality (SSIM)    : {metrics['visual_quality']['ssim']:.4f}")
    print(f"  Signal-to-Noise (PSNR)   : {metrics['visual_quality']['psnr_db']:.2f} dB")
    print(f"  Max Error (L_infinity)   : {metrics['visual_quality']['linf']} px")

    # Commercial API evaluation
    comm = CommercialAPIEvaluator.evaluate_commercial_apis(image_bgr, protected_bgr, pipeline=pipeline)
    print("\n--- Commercial Cloud Face API Verification ---")
    print(f"  AWS Rekognition          : {comm['aws_rekognition']['verdict']} ({comm['aws_rekognition']['similarity_score_pct']}% Similarity)")
    print(f"  Face++ (Megvii)          : {comm['face_plus_plus']['verdict']} ({comm['face_plus_plus']['confidence_score_pct']}% Confidence)")
    print(f"  Azure Face API           : {comm['azure_face']['verdict']} (Confidence: {comm['azure_face']['confidence']:.3f})\n")

    png_bytes = StegoContainer.save_protected_image(protected_bgr, vault_str)
    output_path = args.output if args.output.endswith('.png') else args.output + '.png'
    with open(output_path, 'wb') as f:
        f.write(png_bytes)

    print(f"[+] Protected image saved successfully to: {output_path}")


def decrypt_command(args):
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, 'rb') as f:
        img_bytes = f.read()

    try:
        print("[*] Extracting encrypted biometric vault...")
        vault_str = StegoContainer.extract_vault_from_bytes(img_bytes)
        
        print("[*] Decrypting AES-256-GCM authenticated payload...")
        decrypted = CryptoVault.decrypt_face_payload(vault_str, args.password)
        
        prot_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        restored = StegoContainer.reconstruct_original_image(prot_bgr, decrypted)

        cv2.imwrite(args.output, restored)
        print(f"[+] Original photo losslessly restored to: {args.output} (100% bit-for-bit match)")
    except Exception as e:
        print(f"[-] Decryption failed: {e}")
        sys.exit(1)


def benchmark_command(args):
    sample_dir = "samples"
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_list = []
    for f in sample_files:
        img = cv2.imread(os.path.join(sample_dir, f))
        if img is not None:
            image_list.append({"name": f, "image": img})

    res = CommercialAPIEvaluator.run_open_set_benchmark(
        image_list, protection_method=args.method, passphrase=args.password
    )
    print("\n" + "=" * 70)
    print(" [*] OPEN-SET BENCHMARK & COMMERCIAL API EVALUATION")
    print("=" * 70)
    print(f" Method                     : {res['protection_method']}")
    print(f" Total Identities Evaluated : {res['total_identities_evaluated']}")
    print(f" Protection Success Rate    : {res['protection_success_rate_pct']}% (Target >= 95%: {'MET' if res['is_target_95_plus_met'] else 'NOT MET'})")
    print(f" Average SSIM               : {res['average_ssim']:.4f}")
    print(f" Average PSNR               : {res['average_psnr_db']:.2f} dB")
    print(f" AWS Defeat Rate            : {res['commercial_api_protection_rates']['aws_rekognition_defeat_rate_pct']}%")
    print(f" Face++ Defeat Rate         : {res['commercial_api_protection_rates']['face_plus_plus_defeat_rate_pct']}%")
    print(f" Azure Face Defeat Rate     : {res['commercial_api_protection_rates']['azure_face_defeat_rate_pct']}%\n")


def main():
    parser = argparse.ArgumentParser(description="CYBER MIRAGE: Targeted Identity-Protection (TIP-IM) & Cryptographic Image Privacy CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Protect parser
    p_prot = subparsers.add_parser("protect", help="Protect an image using TIP-IM or keyed cryptographic transformation")
    p_prot.add_argument("--input", "-i", required=True, help="Path to input face image")
    p_prot.add_argument("--output", "-o", default="protected.png", help="Path to output protected PNG image")
    p_prot.add_argument("--password", "-p", required=True, help="Master passphrase for AES-256 encryption")
    p_prot.add_argument("--mode", "-m", default="tip_im", choices=["tip_im", "frequency_dct", "spatial_prf", "reversible_modular"])
    p_prot.add_argument("--target_mode", default="untargeted", choices=["targeted", "untargeted"])
    p_prot.add_argument("--iterations", type=int, default=12, help="Number of TIP-IM iterations")
    p_prot.add_argument("--strength", type=float, default=1.0, help="Transformation strength")
    p_prot.add_argument("--max_linf", type=int, default=6, help="Maximum pixel perturbation bound")

    # Decrypt parser
    p_dec = subparsers.add_parser("decrypt", help="Decrypt and restore original photo from protected image")
    p_dec.add_argument("--input", "-i", required=True, help="Path to protected image")
    p_dec.add_argument("--output", "-o", default="restored.png", help="Path to output restored image")
    p_dec.add_argument("--password", "-p", required=True, help="Master passphrase for AES-256 decryption")

    # Benchmark parser
    p_bench = subparsers.add_parser("benchmark", help="Run open-set benchmark and commercial API verification")
    p_bench.add_argument("--method", default="tip_im", choices=["tip_im", "frequency_dct", "spatial_prf", "reversible_modular"])
    p_bench.add_argument("--password", default="CyberMirage2026!KeyAlpha")

    args = parser.parse_args()
    if args.command == "protect":
        protect_command(args)
    elif args.command == "decrypt":
        decrypt_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)


if __name__ == "__main__":
    main()
