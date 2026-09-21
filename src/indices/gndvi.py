import numpy as np


def calculate_gndvi(green, nir):
    """
    Calculate GNDVI from the green and NIR bands.

    GNDVI = (NIR - Green) / (NIR + Green)
    """

    denominator = nir + green

    gndvi = np.where(
        denominator != 0,
        (nir - green) / denominator,
        np.nan
    )

    return gndvi


if __name__ == "__main__":
    from src.common.sentinel2_data import green, nir

    gndvi = calculate_gndvi(green, nir)

    print("GNDVI calculated successfully!")

    print("\nGNDVI statistics:")
    print("Minimum:", np.nanmin(gndvi))
    print("Maximum:", np.nanmax(gndvi))
    print("Mean:", np.nanmean(gndvi))
    print("Median:", np.nanmedian(gndvi))
    print("Standard deviation:", np.nanstd(gndvi))