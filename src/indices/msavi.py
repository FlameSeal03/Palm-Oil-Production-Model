import numpy as np


def calculate_msavi(red, nir):
    """
    Calculate MSAVI from the red and NIR bands.

    MSAVI = (2*NIR + 1 - sqrt((2*NIR + 1)^2
            - 8*(NIR - Red))) / 2

    MSAVI is designed to reduce the influence of soil background
    while retaining sensitivity to vegetation.
    """

    inside_sqrt = (
        (2 * nir + 1) ** 2
        - 8 * (nir - red)
    )

    # Protect against very small negative floating-point values.
    inside_sqrt = np.maximum(inside_sqrt, 0)

    msavi = (
        2 * nir
        + 1
        - np.sqrt(inside_sqrt)
    ) / 2

    return msavi


if __name__ == "__main__":
    from src.common.sentinel2_data import red, nir

    msavi = calculate_msavi(red, nir)

    print("MSAVI calculated successfully!")
    print("\nMSAVI statistics:")
    print("Minimum:", np.nanmin(msavi))
    print("Maximum:", np.nanmax(msavi))
    print("Mean:", np.nanmean(msavi))
    print("Median:", np.nanmedian(msavi))
    print("Standard deviation:", np.nanstd(msavi))