"""
Cyber Mirage - End-to-End Live API & Integration Verification Script
Tests all live endpoints on http://127.0.0.1:8000
"""

import urllib.request
import urllib.parse
import json
import base64
import numpy as np
import cv2


BASE_URL = "http://127.0.0.1:8000"


def test_health():
    url = f"{BASE_URL}/api/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        data = json.loads(response.read().decode('utf-8'))
        print("[OK] /api/health:", data["status"], data["service"])
        assert data["status"] == "online"


def test_samples():
    url = f"{BASE_URL}/api/samples"
    with urllib.request.urlopen(url) as response:
        assert response.status == 200
        data = json.loads(response.read().decode('utf-8'))
        print(f"[OK] /api/samples: Found {len(data['samples'])} sample portraits:")
        for s in data["samples"]:
            print(f"   - {s['name']}")
        assert len(data["samples"]) >= 3


def test_protect_methods():
    url = f"{BASE_URL}/api/protect_json"
    methods = ["tip_im", "frequency_dct", "spatial_prf", "reversible_modular"]

    for method in methods:
        data_fields = {
            "sample_name": "sample_sarah.jpg",
            "password": "CyberMirage2026!KeyAlpha",
            "method": method,
            "strength": "1.0",
            "max_linf": "6"
        }
        boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
        body = []
        for k, v in data_fields.items():
            body.append(f"--{boundary}".encode('utf-8'))
            body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
            body.append(v.encode('utf-8'))
        body.append(f"--{boundary}--\r\n".encode('utf-8'))
        body_bytes = b"\r\n".join(body)

        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode('utf-8'))
            m = res["metrics"]
            v = m["visual_quality"]
            r = m["recognition_evaluation"]
            p = m["pixel_transformation"]
            print(f"[OK] /api/protect_json [{method}]:")
            print(f"   - SSIM: {v['ssim']:.4f} | PSNR: {v['psnr_db']:.2f} dB | Linf: {v['linf']} | MAE: {v['mae']:.3f}")
            print(f"   - Changed Pixels: {p['changed_pixels_pct']}% ({p['changed_pixels_count']} px)")
            print(f"   - Cosine Dist: {r['cosine_distance']:.4f} | Match Defeated: {not r['ai_match_detected']}")
            assert v["ssim"] >= 0.85
            assert len(res["original_b64"]) > 100
            assert len(res["protected_b64"]) > 100
            assert len(res["diff_map_b64"]) > 100
            assert len(res["download_png_b64"]) > 100


def test_tip_im_optimize_endpoint():
    url = f"{BASE_URL}/api/tip_im_optimize"
    data_fields = {
        "sample_name": "sample_sarah.jpg",
        "password": "CyberMirage2026!KeyAlpha",
        "target_mode": "untargeted",
        "iterations": "8",
        "max_linf": "6"
    }
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    for k, v in data_fields.items():
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
        body.append(v.encode('utf-8'))
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/tip_im_optimize:")
        print(f"   - Iterations Executed: {res['iterations_executed']} in {res['elapsed_ms']}ms")
        print(f"   - Final Cosine Sim: {res['final_cosine_sim']} | Cosine Dist: {res['final_cosine_dist']}")
        print(f"   - Match Defeated: {res['is_defeated']} | SSIM: {res['metrics']['visual_quality']['ssim']:.4f}")
        assert len(res["history"]) == 8
        assert res["final_cosine_dist"] >= 0.50
        assert "commercial_apis" in res


def test_commercial_api_eval():
    url = f"{BASE_URL}/api/commercial_api_eval"
    data_fields = {
        "sample1": "sample_sarah.jpg",
        "sample2": "sample_sarah.jpg"
    }
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    for k, v in data_fields.items():
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
        body.append(v.encode('utf-8'))
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/commercial_api_eval:")
        print(f"   - AWS Rekognition Sim: {res['aws_rekognition']['similarity_score_pct']}% | Match: {res['aws_rekognition']['match_detected']}")
        print(f"   - Face++ Confidence: {res['face_plus_plus']['confidence_score_pct']}% | Match: {res['face_plus_plus']['match_detected']}")
        print(f"   - Azure Face Conf: {res['azure_face']['confidence']} | Identical: {res['azure_face']['is_identical']}")
        assert res["aws_rekognition"]["match_detected"] is True


def test_open_set_benchmark():
    url = f"{BASE_URL}/api/open_set_benchmark?method=tip_im"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/open_set_benchmark:")
        print(f"   - Protection Success Rate: {res['protection_success_rate_pct']}% (Target >= 95%: {res['is_target_95_plus_met']})")
        print(f"   - Average SSIM: {res['average_ssim']} | PSNR: {res['average_psnr_db']} dB")
        assert res["protection_success_rate_pct"] >= 95.0


def test_key_sensitivity():
    url = f"{BASE_URL}/api/key_sensitivity_json"
    data_fields = {
        "sample_name": "sample_sarah.jpg",
        "password_k1": "CyberMirage2026!KeyAlpha",
        "password_k2": "CyberMirage2026!KeyAlphb",
        "method": "frequency_dct"
    }
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    for k, v in data_fields.items():
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
        body.append(v.encode('utf-8'))
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/key_sensitivity_json (1-Bit Flip):")
        print(f"   - Determinism Error: {res['determinism_error']} (Is Deterministic: {res['is_deterministic']})")
        print(f"   - Divergence Changed Pixels: {res['key_separation_changed_pixels_pct']}% | MAE: {res['key_separation_mae']} px")
        assert res["is_deterministic"] is True
        assert res["key_separation_changed_pixels_pct"] > 10.0


def test_compare_linkage():
    url = f"{BASE_URL}/api/compare_json"
    data_fields = {
        "sample1": "sample_sarah.jpg",
        "sample2": "sample_sarah.jpg",
        "protect_photo2": "true",
        "method": "frequency_dct",
        "strength": "1.0",
        "password": "CyberMirage2026!KeyAlpha"
    }
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    for k, v in data_fields.items():
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
        body.append(v.encode('utf-8'))
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/compare_json (Protected Linkage):")
        print(f"   - Cosine Distance: {res['cosine_distance']:.4f} | Cosine Sim: {res['cosine_similarity']:.4f}")
        print(f"   - AI Match Detected: {res['ai_match_detected']} | Confidence: {res['ai_match_confidence']:.1f}%")
        assert "cosine_distance" in res
        assert "photo1_b64" in res
        assert "photo2_b64" in res


def test_decrypt_vault():
    protect_url = f"{BASE_URL}/api/protect_json"
    data_fields = {
        "sample_name": "sample_sarah.jpg",
        "password": "CyberMirage2026!KeyAlpha",
        "method": "frequency_dct"
    }
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    for k, v in data_fields.items():
        body.append(f"--{boundary}".encode('utf-8'))
        body.append(f'Content-Disposition: form-data; name="{k}"\r\n'.encode('utf-8'))
        body.append(v.encode('utf-8'))
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        protect_url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        prot_res = json.loads(resp.read().decode('utf-8'))
        png_b64 = prot_res["download_png_b64"].split(",")[1]
        png_bytes = base64.b64decode(png_b64)

    decrypt_url = f"{BASE_URL}/api/decrypt_json"
    boundary = "----CyberMirageBoundary7MA4YWxkTrZu0gW"
    body = []
    body.append(f"--{boundary}".encode('utf-8'))
    body.append(b'Content-Disposition: form-data; name="file"; filename="protected.png"\r\nContent-Type: image/png\r\n\r\n' + png_bytes)
    body.append(f"--{boundary}".encode('utf-8'))
    body.append(b'Content-Disposition: form-data; name="password"\r\n\r\nCyberMirage2026!KeyAlpha')
    body.append(f"--{boundary}--\r\n".encode('utf-8'))
    body_bytes = b"\r\n".join(body)

    req = urllib.request.Request(
        decrypt_url,
        data=body_bytes,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        dec_res = json.loads(resp.read().decode('utf-8'))
        print("[OK] /api/decrypt_json:")
        print(f"   - Status: {dec_res['status']}")
        print(f"   - Fidelity: {dec_res['pixel_fidelity']}")
        print(f"   - Restored SSIM: {dec_res['restored_ssim']}")
        assert dec_res["restored_ssim"] == 1.0


if __name__ == "__main__":
    print("=" * 60)
    print("CYBER MIRAGE -- LIVE API VERIFICATION")
    print("=" * 60)
    test_health()
    test_samples()
    test_protect_methods()
    test_tip_im_optimize_endpoint()
    test_commercial_api_eval()
    test_open_set_benchmark()
    test_key_sensitivity()
    test_compare_linkage()
    test_decrypt_vault()
    print("=" * 60)
    print("[PASS] ALL LIVE INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 60)
