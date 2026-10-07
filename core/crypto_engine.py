"""
Cyber Mirage - Cryptographic Key Derivation & Secure Keystream Engine
Implements:
  - PBKDF2-HMAC-SHA256 key derivation
  - AES-CTR / ChaCha20 secure pseudorandom keystream generator
  - AES-256-GCM authenticated cryptographic vault for lossless biometric recovery
"""

import os
import json
import base64
import hashlib
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


class KeyDerivationEngine:
    """
    Handles cryptographic key derivation from user secret passphrases.
    Never logs or exposes the raw secret key.
    """
    SALT_SIZE = 16
    ITERATIONS = 100_000
    KEY_LENGTH = 32  # 256 bits

    @classmethod
    def derive_key(cls, passphrase: str, salt: bytes = None) -> tuple[bytes, bytes]:
        """
        Derives a 256-bit cryptographic key using PBKDF2-HMAC-SHA256.
        Returns (derived_key, salt).
        """
        if salt is None:
            # Deterministic salt derived from passphrase hash for reproducible transformations
            salt = hashlib.sha256(passphrase.encode('utf-8')).digest()[:cls.SALT_SIZE]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=cls.KEY_LENGTH,
            salt=salt,
            iterations=cls.ITERATIONS
        )
        derived_key = kdf.derive(passphrase.encode('utf-8'))
        return derived_key, salt


class CryptoStreamGenerator:
    """
    Generates deterministic, cryptographically secure pseudo-random keystreams
    using AES-CTR / ChaCha20 for pixel and frequency coefficient modulation.
    """

    @classmethod
    def generate_keystream_float(
        cls,
        key: bytes,
        shape: tuple,
        nonce: bytes = None,
        low: float = -1.0,
        high: float = 1.0
    ) -> np.ndarray:
        """
        Generates a numpy float array of given shape with values uniformly
        distributed in [low, high], cryptographically derived from the key and nonce.
        """
        total_elements = int(np.prod(shape))
        num_bytes = total_elements * 4  # 4 bytes (uint32) per element

        if nonce is None:
            nonce = hashlib.sha256(key + b"cyber_mirage_stream_v2").digest()[:16]

        # Use AES-CTR stream cipher to produce cryptographically secure bytes
        cipher = Cipher(algorithms.AES(key), modes.CTR(nonce))
        encryptor = cipher.encryptor()
        
        # Encrypt zero blocks to produce raw pseudo-random stream
        zero_blocks = b"\x00" * num_bytes
        keystream_bytes = encryptor.update(zero_blocks) + encryptor.finalize()

        # Convert to uint32 numpy array
        raw_uint32 = np.frombuffer(keystream_bytes, dtype=np.uint32)
        
        # Normalize to [0.0, 1.0)
        normalized = raw_uint32.astype(np.float64) / 4294967296.0
        
        # Scale to [low, high]
        scaled = low + normalized * (high - low)
        return scaled.reshape(shape)

    @classmethod
    def generate_keystream_int(
        cls,
        key: bytes,
        shape: tuple,
        nonce: bytes = None,
        min_val: int = -4,
        max_val: int = 4
    ) -> np.ndarray:
        """
        Generates bounded integer transformation deltas in [min_val, max_val].
        """
        float_stream = cls.generate_keystream_float(key, shape, nonce, low=0.0, high=1.0)
        range_span = max_val - min_val + 1
        int_stream = np.floor(float_stream * range_span).astype(np.int32) + min_val
        return np.clip(int_stream, min_val, max_val)


class CryptoVault:
    """
    Handles authenticated AES-256-GCM encryption/decryption of raw biometric data
    for 100% exact mathematical restoration.
    """
    SALT_SIZE = 16
    NONCE_SIZE = 12

    @classmethod
    def encrypt_face_payload(cls, face_crop: np.ndarray, bbox: list, landmarks: list, passphrase: str) -> str:
        success, encoded = cv2.imencode('.png', face_crop)
        if not success:
            raise ValueError("Failed to encode face patch to PNG")

        raw_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        payload = {
            "bbox": [int(v) for v in bbox],
            "landmarks": [[float(pt[0]), float(pt[1])] for pt in landmarks] if landmarks is not None else [],
            "face_data_b64": raw_b64,
            "shape": list(face_crop.shape)
        }
        plaintext = json.dumps(payload).encode('utf-8')

        salt = os.urandom(cls.SALT_SIZE)
        nonce = os.urandom(cls.NONCE_SIZE)
        key, _ = KeyDerivationEngine.derive_key(passphrase, salt)

        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=b"cyber_mirage_v2")
        packed = salt + nonce + ciphertext
        return base64.b64encode(packed).decode('utf-8')

    @classmethod
    def decrypt_face_payload(cls, vault_b64: str, passphrase: str) -> dict:
        try:
            packed = base64.b64decode(vault_b64.strip())
            salt = packed[:cls.SALT_SIZE]
            nonce = packed[cls.SALT_SIZE:cls.SALT_SIZE + cls.NONCE_SIZE]
            ciphertext = packed[cls.SALT_SIZE + cls.NONCE_SIZE:]

            key, _ = KeyDerivationEngine.derive_key(passphrase, salt)
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=b"cyber_mirage_v2")
            payload = json.loads(plaintext.decode('utf-8'))

            face_bytes = base64.b64decode(payload["face_data_b64"])
            np_arr = np.frombuffer(face_bytes, dtype=np.uint8)
            restored_face = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            return {
                "face_crop": restored_face,
                "bbox": payload["bbox"],
                "landmarks": payload.get("landmarks", []),
                "shape": payload["shape"]
            }
        except Exception as e:
            raise ValueError(f"Authentication/Decryption failed: {str(e)}")
