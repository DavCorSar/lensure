"""
This module implements the definition of the authority that
will sign the images.
"""

import struct

from PIL import Image
import numpy as np
import imagehash
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import reedsolo
import pywt

VALID_EMBED_METHODS = ["LSB", "DWT"]


class Authority:
    """
    Implementation of a trusted authority
    """

    def __init__(
        self,
        embed_og_hash: bool,
        public_exponent: int = 65537,
        key_size: int = 2048,
        embed_method: str = "LSB",
    ):
        self.embed_og_hash = embed_og_hash
        self._private_key, self.public_key = self.generate_keys(
            public_exponent, key_size
        )
        if embed_method not in VALID_EMBED_METHODS:
            raise ValueError(f"{embed_method} is not a valid method")
        self.embed_method = embed_method

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
        Checks if the signature corresponds with the specifyied hash
        """
        h = np.asarray(list(h), dtype=np.uint8).flatten()
        h_bytes = np.packbits(h).tobytes()

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
            hash_to_insert = "".join([str(i) for i in h])
            m = f"{hash_to_insert}|{sig.hex()}"
        else:
            m = f"{sig.hex()}"

        encoded = self._encode_message(m)

        if self.embed_method == "LSB":
            return self._embed_lsb(image, encoded)
        if self.embed_method == "DWT":
            return self._embed_dwt(image, encoded)
        raise ValueError(f"Embed method {self.embed_method} is not valid")

    def extract_watermark(self, image: Image) -> dict:
        """
        Extracts the signature based on the specifyied method
        """
        if self.embed_method == "LSB":
            return self._extract_watermark_lsb(image)
        if self.embed_method == "DWT":
            return self._extract_watermark_dwt(image)
        raise ValueError(f"Embed method {self.embed_method} is not valid")

    def _extract_watermark_lsb(self, image: Image) -> dict:
        """
        Extracts the signature and the original hash
        """
        img = np.array(image)
        flat = img.flatten()

        header_bits = flat[:32] & 1

        header_bytes = np.packbits(header_bits).tobytes()

        length = struct.unpack("I", header_bytes)[0]

        total_bits = (length + 4) * 8
        bits = flat[:total_bits] & 1
        bytes_data = np.packbits(bits).tobytes()

        payload = bytes_data[4:]

        rs = reedsolo.RSCodec(10)
        decoded = rs.decode(payload)[0].decode()

        if self.embed_og_hash:
            h, sig = decoded.split("|")
            return {
                "hash": np.array([int(i) for i in h]),
                "signature": bytes.fromhex(sig),
            }

        return {"signature": bytes.fromhex(decoded)}

    def _extract_watermark_dwt(self, image: Image, delta: float = 20.0) -> dict:
        """
        Extracts the watermark payload from the image using DWT-domain extraction.
        """
        header_bits = self._extract_dwt_bits(image, 32, delta=delta)
        header_bytes = np.packbits(header_bits).tobytes()

        length = struct.unpack("I", header_bytes)[0]

        total_bits = (length + 4) * 8
        bits = self._extract_dwt_bits(image, total_bits, delta=delta)
        bytes_data = np.packbits(bits).tobytes()

        payload = bytes_data[4:]

        rs = reedsolo.RSCodec(10)
        decoded = rs.decode(payload)[0].decode()

        if self.embed_og_hash:
            h, sig = decoded.split("|")
            return {
                "hash": np.array([int(i) for i in h]),
                "signature": bytes.fromhex(sig),
            }

        return {"signature": bytes.fromhex(decoded)}

    def _encode_message(self, message: str) -> bytes:
        """
        Encodes message with Reed-Solomon + length header
        """

        rs = reedsolo.RSCodec(10)

        raw = message.encode()
        encoded = rs.encode(raw)

        # añadimos header con longitud
        length = len(encoded)
        header = struct.pack("I", length)

        return header + encoded

    def _decode_reed_solomon(self, data: bytes) -> str:
        """
        Decodes the message using Reed-Solomon
        """
        rs = reedsolo.RSCodec(10)
        decoded = rs.decode(data)[0]
        return decoded.decode()

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

    def _embed_dwt(self, image: Image, data: bytes, delta: float = 20.0) -> Image:
        """
        Embeds the message using a DWT-domain QIM watermarking scheme.
        """
        channels = image.convert("YCbCr").split()
        y_array = np.array(channels[0]).astype(np.float32)
        shape = y_array.shape
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        y_array = self._pad_even_shape(y_array)
        coeffs = self._embed_bits_in_dwt_coeffs(y_array, bits, delta)
        marked_y = pywt.idwt2(coeffs, "haar", mode="periodization")
        marked_y = self._to_uint8_image_array(marked_y, shape)

        return Image.merge(
            "YCbCr",
            (Image.fromarray(marked_y, mode="L"), channels[1], channels[2]),
        ).convert("RGB")

    def _extract_dwt_bits(
        self, image: Image, num_bits: int, delta: float = 20.0
    ) -> np.ndarray:
        """
        Extracts num_bits from the DWT-domain watermark using QIM parity decoding.
        """
        y_array = np.array(image.convert("YCbCr").split()[0]).astype(np.float32)
        coeff_vector = self._get_dwt_embedding_coefficients(
            self._pad_even_shape(y_array)
        )

        if num_bits > coeff_vector.size:
            raise ValueError(f"Not enough DWT coefficients to extract {num_bits} bits.")

        return self._extract_bits_qim(coeff_vector[:num_bits], delta)

    def _embed_bits_in_dwt_coeffs(
        self,
        y_array: np.ndarray,
        bits: np.ndarray,
        delta: float,
    ) -> tuple:
        """
        Embeds bits into the LH and HL DWT subbands.
        """
        approximation, details = pywt.dwt2(y_array, "haar", mode="periodization")
        lh_coeffs, hl_coeffs, hh_coeffs = details

        coeff_vector = np.concatenate((lh_coeffs.ravel(), hl_coeffs.ravel()))

        if bits.size > coeff_vector.size:
            raise ValueError(
                f"Image is too small for the DWT watermarking. "
                f"Required bits: {bits.size}, capacity: {coeff_vector.size}"
            )

        coeff_vector[: bits.size] = self._embed_bits_qim(
            coeff_vector[: bits.size],
            bits,
            delta,
        )

        return (
            approximation,
            (
                coeff_vector[: lh_coeffs.size].reshape(lh_coeffs.shape),
                coeff_vector[lh_coeffs.size :].reshape(hl_coeffs.shape),
                hh_coeffs,
            ),
        )

    @staticmethod
    def compute_perceptual_hash(image: Image) -> np.ndarray:
        """
        Returns the perceptual hash of the image
        """
        img = image.convert("L")
        h = imagehash.whash(img)
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
        coefficients: np.ndarray,
        bits: np.ndarray,
        delta: float,
    ) -> np.ndarray:
        """
        Embeds bits into coefficients using parity-based QIM.
        """
        quantized = np.round(coefficients / delta).astype(int)
        mismatch = (quantized % 2) != bits

        quantized[mismatch] += np.where(
            coefficients[mismatch] >= 0,
            1,
            -1,
        )

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
        """
        return (np.round(coefficients / delta).astype(int) % 2).astype(np.uint8)
