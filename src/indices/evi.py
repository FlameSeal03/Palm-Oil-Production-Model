import numpy as np


def calculate_evi(blue, red, nir):
    """
    Calculate EVI from the blue, red, and NIR bands.

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    """

    denominator = nir + 6 * red - 7.5 * blue + 1

    evi = np.where(
        denominator != 0,
        2.5 * (nir - red) / denominator,
        np.nan
    )

    return evi


if __name__ == "__main__":
    from src.common.sentinel2_data import blue, red, nir

    evi = calculate_evi(blue, red, nir)

    print("EVI calculated successfully!")

    print("\nEVI statistics:")
    print("Minimum:", np.nanmin(evi))
    print("Maximum:", np.nanmax(evi))
    print("Mean:", np.nanmean(evi))
    print("Median:", np.nanmedian(evi))
    print("Standard deviation:", np.nanstd(evi))