import numpy as np


def calculate_savi(red, nir, L=0.5):
    """
    Calculate SAVI from the red and NIR bands.

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    L is the soil brightness correction factor.
    A value of 0.5 is commonly used for intermediate vegetation cover.
    """

    denominator = nir + red + L

    savi = np.where(
        denominator != 0,
        ((nir - red) / denominator) * (1 + L),
        np.nan
    )

    return savi


if __name__ == "__main__":
    from src.common.sentinel2_data import red, nir

    savi = calculate_savi(red, nir)

    print("SAVI calculated successfully!")
    print("\nSAVI statistics:")
    print("Minimum:", np.nanmin(savi))
    print("Maximum:", np.nanmax(savi))
    print("Mean:", np.nanmean(savi))
    print("Median:", np.nanmedian(savi))
    print("Standard deviation:", np.nanstd(savi))