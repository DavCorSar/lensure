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


class Authority:
    """
    Implementation of a trusted authority
    """

    def __init__(
        self, embed_og_hash: bool, public_exponent: int = 65537, key_size: int = 2048
    ):
        self.embed_og_hash = embed_og_hash
        self._private_key, self.public_key = self.generate_keys(
            public_exponent, key_size
        )

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
        h = np.asarray([i for i in h], dtype=np.uint8).flatten()
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
        except Exception as e:
            print("Exception when could not verify is: ", e)
            return False

    def include_watermarking(self, image: Image) -> Image:
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

        watermarked = self._embed_lsb(image, encoded)

        return watermarked

    def extract_watermark(self, image: Image) -> dict:
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

    @staticmethod
    def compute_perceptual_hash(image: Image) -> np.ndarray:
        """
        Returns the perceptual hash of the image
        """
        img = image.convert("L")
        h = imagehash.whash(img)
        return h.hash.astype(np.uint8).flatten()
