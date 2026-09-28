import numpy as np


def calculate_palm_mask(
    ndvi,
    ndre,
    ndvi_threshold=0.60,
    ndre_threshold=0.40,
):
    """
    Create a binary palm-tree candidate mask.

    A pixel is classified as a palm candidate when
    it meets both the NDVI and NDRE thresholds.
    """

    palm_mask = (
        (ndvi >= ndvi_threshold)
        & (ndre >= ndre_threshold)
    )

    return palm_mask.astype(np.uint8)