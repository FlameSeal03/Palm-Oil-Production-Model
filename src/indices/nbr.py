import numpy as np


def calculate_nbr(nir, swir_1):
    """
    Calculate NBR from the NIR and SWIR bands.

    NBR = (NIR - SWIR) / (NIR + SWIR)

    NBR is commonly used to characterize vegetation disturbance
    and canopy condition.
    """

    denominator = nir + swir_1

    nbr = np.where(
        denominator != 0,
        (nir - swir_1) / denominator,
        np.nan
    )

    return nbr


if __name__ == "__main__":
    from src.common.sentinel2_data import nir, swir_1

    nbr = calculate_nbr(nir, swir_1)

    print("NBR calculated successfully!")
    print("\nNBR statistics:")
    print("Minimum:", np.nanmin(nbr))
    print("Maximum:", np.nanmax(nbr))
    print("Mean:", np.nanmean(nbr))
    print("Median:", np.nanmedian(nbr))
    print("Standard deviation:", np.nanstd(nbr))