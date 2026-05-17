"""
This module implements the definition of the authority that
will sign the images.
"""

from PIL import Image
import numpy as np
import imagehash
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import reedsolo
import pywt

VALID_EMBED_METHODS = ["LSB", "DWT"]
VALID_HASH_TYPES = ["whash", "phash", "dhash", "ahash"]

# The Y channel is normalised to this size before DWT embedding/extraction so
# that the coefficient positions are invariant to the input image resolution.
# Capacity at 512×512: (256×256)×2 = 131 072 coefficients > 70 784 bits needed.
_DWT_FIXED_SIZE = (512, 512)  # (width, height) in PIL convention


class Authority:
    """
    Implementation of a trusted authority
    """

    def __init__(
        self,
        embed_og_hash: bool,
        public_exponent: int = 65537,
        key_size: int = 2048,
        embed_method: str = "DWT",
        delta_dwt: float = 40.0,
        hash_size: int = 8,
        hash_type: str = "whash",
    ):
        self.embed_og_hash = embed_og_hash
        self._private_key, self.public_key = self.generate_keys(
            public_exponent, key_size
        )
        if embed_method not in VALID_EMBED_METHODS:
            raise ValueError(f"{embed_method} is not a valid method")
        if hash_type not in VALID_HASH_TYPES:
            raise ValueError(
                f"{hash_type} is not a valid hash type. Choose from {VALID_HASH_TYPES}"
            )
        self.embed_method = embed_method
        self.delta_dwt = delta_dwt
        self.dwt_repetition = 7
        self.hash_size = hash_size
        self.hash_type = hash_type

    def generate_keys(
        self, public_exponent: int, key_size: int
    ) -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """
        Generates a public and private key
        """
        sk = rsa.generate_private_key(
            public_exponent=public_exponent, key_size=key_size
        )
        pk = sk.public_key()
        return sk, pk

    def sign_hash(self, hash_to_sign: np.ndarray):
        """
        Returns the signed hash using the private key
        """
        h_bytes = np.packbits(hash_to_sign).tobytes()

        return self._private_key.sign(
            h_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

    def verify_signature(self, h: str, signature: bytes) -> bool:
        """
        Checks if the signature corresponds with the specified hash
        """
        if signature == b"":
            return False

        if isinstance(h, str):
            h_arr = np.array([int(i) for i in h], dtype=np.uint8)
        else:
            h_arr = np.asarray(h, dtype=np.uint8).flatten()

        h_bytes = np.packbits(h_arr).tobytes()

        try:
            self.public_key.verify(
                signature,
                h_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False

    def embed_watermark(self, image: Image) -> Image:
        """
        Generates a new image that includes the watermarking
        """
        h = self.compute_perceptual_hash(image)

        sig = self.sign_hash(h)

        if self.embed_og_hash:
            hash_bytes = np.packbits(h).tobytes()
            m = hash_bytes + sig
        else:
            m = sig

        encoded = self._encode_message(m)

        if self.embed_method == "LSB":
            return self._embed_lsb(image, encoded)
        if self.embed_method == "DWT":
            return self._embed_dwt(image, encoded)
        raise ValueError(f"Embed method {self.embed_method} is not valid")

    def extract_watermark(self, image: Image) -> dict:
        """
        Extracts the signature based on the specified method
        """
        if self.embed_method == "LSB":
            return self._extract_watermark_lsb(image)
        if self.embed_method == "DWT":
            return self._extract_watermark_dwt(image)
        raise ValueError(f"Embed method {self.embed_method} is not valid")

    def _extract_watermark_lsb(self, image: Image) -> dict:
        """
        Extracts the signature and the original hash without using headers.
        """
        img = np.array(image)
        flat = img.flatten()

        expected_len = self._expected_encoded_length()
        total_bits = expected_len * 8

        if total_bits > flat.size:
            return {
                "hash": np.zeros(self._hash_bits, dtype=np.uint8),
                "signature": b"",
                "decode_error": (
                    f"Image too small for LSB extraction. "
                    f"Required bits: {total_bits}, capacity: {flat.size}"
                ),
            }

        bits = flat[:total_bits] & 1
        payload = np.packbits(bits).tobytes()

        return self._decode_payload(payload)

    def _extract_watermark_dwt(self, image: Image) -> dict:
        """
        Extracts the watermark payload from the image using DWT-domain extraction.
        """
        expected_len = self._expected_encoded_length()
        payload_bits = expected_len * 8
        total_bits = payload_bits * self.dwt_repetition

        try:
            raw_bits = self._extract_dwt_bits(image, total_bits)
        except ValueError as e:
            return {
                "hash": np.zeros(self._hash_bits, dtype=np.uint8),
                "signature": b"",
                "decode_error": str(e),
            }

        if self.dwt_repetition > 1:
            repeated = raw_bits.reshape(self.dwt_repetition, payload_bits)
            bits = (repeated.sum(axis=0) >= (self.dwt_repetition // 2 + 1)).astype(
                np.uint8
            )
        else:
            bits = raw_bits
        payload = np.packbits(bits).tobytes()

        return self._decode_payload(payload)

    def _encode_message(self, message: bytes) -> bytes:
        """
        Encodes a binary message with Reed-Solomon
        """

        rs = reedsolo.RSCodec(200)

        return rs.encode(message)

    def _embed_lsb(self, image: Image, data: bytes) -> Image:
        """
        Inserts the message in the LSB of the image
        """
        img = np.array(image)
        flat = img.flatten()

        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        if len(bits) > len(flat):
            raise ValueError("Image is too small for the watermarking")

        flat[: len(bits)] &= 0xFE
        flat[: len(bits)] |= bits

        img_watermarked = flat.reshape(img.shape)
        return Image.fromarray(img_watermarked.astype(np.uint8))

    def _embed_dwt(self, image: Image, data: bytes) -> Image:
        """
        Embeds the message using a DWT-domain QIM watermarking scheme in YCbCr space.
        The Y channel is normalised to _DWT_FIXED_SIZE before the DWT so that
        embedding positions are invariant to the input resolution.
        """
        channels = image.convert("YCbCr").split()
        y = channels[0]

        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        if self.dwt_repetition > 1:
            bits = np.tile(bits, self.dwt_repetition)

        y_fixed = np.array(y.resize(_DWT_FIXED_SIZE, Image.LANCZOS)).astype(np.float32)
        y_padded = self._pad_even_shape(y_fixed)
        coeffs = self._embed_bits_in_dwt_coeffs(y_padded, bits)
        marked_y_fixed = pywt.idwt2(coeffs, "haar", mode="periodization")
        marked_y_fixed = self._to_uint8_image_array(marked_y_fixed, y_fixed.shape)

        marked_y = Image.fromarray(marked_y_fixed, mode="L").resize(
            y.size, Image.LANCZOS
        )
        return Image.merge("YCbCr", (marked_y, channels[1], channels[2])).convert("RGB")

    def _extract_dwt_bits(self, image: Image, num_bits: int) -> np.ndarray:
        """
        Extracts num_bits from the DWT-domain watermark using QIM parity decoding.
        """
        y = image.convert("YCbCr").split()[0]
        y_fixed = np.array(y.resize(_DWT_FIXED_SIZE, Image.LANCZOS)).astype(np.float32)
        coeff_vector = self._get_dwt_embedding_coefficients(
            self._pad_even_shape(y_fixed)
        )

        if num_bits > coeff_vector.size:
            raise ValueError(f"Not enough DWT coefficients to extract {num_bits} bits.")

        positions = self._get_dwt_positions(coeff_vector.size, num_bits)

        return self._extract_bits_qim(coeff_vector[positions], self.delta_dwt)

    def _embed_bits_in_dwt_coeffs(
        self,
        y_array: np.ndarray,
        bits: np.ndarray,
    ) -> tuple:
        """
        Embeds bits into the LH and HL DWT subbands.
        """
        approximation, details = pywt.dwt2(y_array, "haar", mode="periodization")
        lh_coeffs, hl_coeffs, hh_coeffs = details

        coeff_vector = np.concatenate((lh_coeffs.ravel(), hl_coeffs.ravel()))

        positions = self._get_dwt_positions(coeff_vector.size, bits.size)

        coeff_vector[positions] = self._embed_bits_qim(
            coeff_vector[positions], bits, self.delta_dwt
        )

        return (
            approximation,
            (
                coeff_vector[: lh_coeffs.size].reshape(lh_coeffs.shape),
                coeff_vector[lh_coeffs.size :].reshape(hl_coeffs.shape),
                hh_coeffs,
            ),
        )

    def _expected_encoded_length(self) -> int:
        """
        Returns the expected encoded payload length in bytes.
        """
        sig_len = self.public_key.key_size // 8

        if self.embed_og_hash:
            raw_len = self._hash_bytes + sig_len
        else:
            raw_len = sig_len

        rs = reedsolo.RSCodec(200)
        dummy = b"\x00" * raw_len
        return len(rs.encode(dummy))

    def _decode_payload(self, payload: bytes) -> dict:
        """
        Decodes a Reed-Solomon protected binary payload.
        """
        rs = reedsolo.RSCodec(200)

        try:
            decoded = rs.decode(payload)[0]
        except reedsolo.ReedSolomonError as e:
            return {
                "hash": np.zeros(self._hash_bits, dtype=np.uint8),
                "signature": b"",
                "decode_error": f"Reed-Solomon failed: {e}",
            }

        sig_len = self.public_key.key_size // 8

        if self.embed_og_hash:
            hash_bytes = decoded[: self._hash_bytes]
            signature = decoded[self._hash_bytes : self._hash_bytes + sig_len]

            if len(hash_bytes) != self._hash_bytes:
                return {
                    "hash": np.zeros(self._hash_bits, dtype=np.uint8),
                    "signature": b"",
                    "decode_error": f"Invalid hash length: {len(hash_bytes)}",
                }

            if len(signature) != sig_len:
                return {
                    "hash": np.zeros(self._hash_bits, dtype=np.uint8),
                    "signature": b"",
                    "decode_error": (
                        f"Invalid signature length: {len(signature)}, "
                        f"expected {sig_len}"
                    ),
                }

            h = np.unpackbits(np.frombuffer(hash_bytes, dtype=np.uint8))

            return {
                "hash": h.astype(np.uint8),
                "signature": signature,
            }

        signature = decoded[:sig_len]

        if len(signature) != sig_len:
            return {
                "signature": b"",
                "decode_error": (
                    f"Invalid signature length: {len(signature)}, expected {sig_len}"
                ),
            }

        return {"signature": signature}

    def _get_dwt_positions(self, capacity: int, num_bits: int) -> np.ndarray:
        """
        Returns deterministic pseudo-random coefficient positions for DWT embedding.
        """
        if num_bits > capacity:
            raise ValueError(
                f"Not enough DWT coefficients. Required bits: {num_bits}, "
                f"capacity: {capacity}"
            )

        rng = np.random.default_rng(42)
        return rng.permutation(capacity)[:num_bits]

    @property
    def default_threshold(self) -> int:
        """Threshold for Hamming distance, auto-scaled with hash_size."""
        return max(1, round(3 * (self.hash_size**2) / 64))

    @property
    def _hash_bits(self) -> int:
        return self.hash_size**2

    @property
    def _hash_bytes(self) -> int:
        return self._hash_bits // 8

    def compute_perceptual_hash(self, image: Image) -> np.ndarray:
        """
        Returns the perceptual hash of the image using the configured hash type and size.
        """
        img = image.convert("L")
        if self.hash_type == "whash":
            h = imagehash.whash(img, hash_size=self.hash_size)
        elif self.hash_type == "phash":
            h = imagehash.phash(img, hash_size=self.hash_size)
        elif self.hash_type == "dhash":
            h = imagehash.dhash(img, hash_size=self.hash_size)
        elif self.hash_type == "ahash":
            h = imagehash.average_hash(img, hash_size=self.hash_size)
        else:
            raise ValueError(f"Unknown hash type: {self.hash_type}")
        return h.hash.astype(np.uint8).flatten()

    @staticmethod
    def _pad_even_shape(array: np.ndarray) -> np.ndarray:
        """
        Pads an array to ensure that both dimensions are even.
        """
        return np.pad(
            array,
            ((0, array.shape[0] % 2), (0, array.shape[1] % 2)),
            mode="edge",
        )

    @staticmethod
    def _to_uint8_image_array(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """
        Crops an array to the original shape and converts it to uint8.
        """
        return np.clip(
            array[: shape[0], : shape[1]],
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _embed_bits_qim(
        coefficients: np.ndarray, bits: np.ndarray, delta: float
    ) -> np.ndarray:
        """
        Embeds bits into coefficients using parity-based QIM.
        """
        quantized = np.round(coefficients / delta).astype(int)
        mismatch = (quantized % 2) != bits
        r = coefficients - quantized * delta
        quantized[mismatch] += np.where(r[mismatch] >= 0, 1, -1)
        return quantized * delta

    @staticmethod
    def _get_dwt_embedding_coefficients(y_array: np.ndarray) -> np.ndarray:
        """
        Returns the concatenated LH and HL coefficients used for embedding.
        """
        _, details = pywt.dwt2(y_array, "haar", mode="periodization")
        lh_coeffs, hl_coeffs, _ = details
        return np.concatenate((lh_coeffs.ravel(), hl_coeffs.ravel()))

    @staticmethod
    def _extract_bits_qim(coefficients: np.ndarray, delta: float) -> np.ndarray:
        """
        Extracts bits from coefficients using parity-based QIM decoding.
        Uses round-half-away-from-zero so that coefficients at exactly ±delta/2
        (caused by uint8 boundary clipping) round to the correct parity.
        """
        return (np.floor(np.abs(coefficients / delta) + 0.5).astype(int) % 2).astype(
            np.uint8
        )
