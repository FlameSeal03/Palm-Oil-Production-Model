import numpy as np


def calculate_ndvi(red, nir):
    """
    Calculate NDVI from red and NIR bands.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    denominator = nir + red

    ndvi = np.where(
        denominator != 0,
        (nir - red) / denominator,
        np.nan
    )

    return ndvi


if __name__ == "__main__":
    from src.common.sentinel2_data import red, nir

    ndvi = calculate_ndvi(red, nir)

    print("NDVI calculated successfully!")

    print("\nNDVI statistics:")
    print("Minimum:", np.nanmin(ndvi))
    print("Maximum:", np.nanmax(ndvi))
    print("Mean:", np.nanmean(ndvi))
    print("Median:", np.nanmedian(ndvi))
    print("Standard deviation:", np.nanstd(ndvi))