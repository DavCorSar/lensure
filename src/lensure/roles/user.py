"""
This module defines the actions a user must do to check if an image is original or not
"""

from PIL import Image
import numpy as np

from lensure.roles.authority import Authority


class User:
    """
    Definition of the user that interacts with the image
    and wants to ensure that it is original
    """

    def __init__(self, authority: Authority):
        self.authority = authority

    def verify(
        self, attacked_image: Image, watermarked_image: Image, threshold=10
    ) -> dict:
        """
        The user has access to the CA at any time to validate the image
        """
        h_new = Authority.compute_perceptual_hash(attacked_image)
        result = self.authority.extract_watermark(watermarked_image)

        if self.authority.embed_og_hash:
            valid_sig = self.authority.verify_signature(
                result["hash"], result["signature"]
            )
            distance = np.sum(result["hash"] != h_new)
        else:
            valid_sig = self.authority.verify_signature(h_new, result["signature"])
            distance = 0

        return {
            "signature_valid": valid_sig,
            "distance": distance,
            "accepted": valid_sig and distance < threshold,
        }
