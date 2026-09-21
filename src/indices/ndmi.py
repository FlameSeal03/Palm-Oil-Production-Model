import numpy as np


def calculate_ndmi(nir, swir_1):
    """
    Calculate NDMI from the NIR and SWIR bands.

    NDMI = (NIR - SWIR) / (NIR + SWIR)
    """

    denominator = nir + swir_1

    ndmi = np.where(
        denominator != 0,
        (nir - swir_1) / denominator,
        np.nan
    )

    return ndmi


if __name__ == "__main__":
    from src.common.sentinel2_data import nir, swir_1

    ndmi = calculate_ndmi(nir, swir_1)

    print("NDMI calculated successfully!")

    print("\nNDMI statistics:")
    print("Minimum:", np.nanmin(ndmi))
    print("Maximum:", np.nanmax(ndmi))
    print("Mean:", np.nanmean(ndmi))
    print("Median:", np.nanmedian(ndmi))
    print("Standard deviation:", np.nanstd(ndmi))