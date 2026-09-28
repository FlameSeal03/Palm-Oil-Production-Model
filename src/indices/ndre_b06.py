import numpy as np


def calculate_ndre_b06(red_edge_2, nir_2):
    """
    Calculate an NDRE-style index using Sentinel-2 B06 and B8A.

    NDRE-B06 = (B8A - B06) / (B8A + B06)
    """

    denominator = nir_2 + red_edge_2

    ndre_b06 = np.where(
        denominator != 0,
        (nir_2 - red_edge_2) / denominator,
        np.nan
    )

    return ndre_b06


if __name__ == "__main__":
    from src.common.sentinel2_data import red_edge_2, nir_2

    ndre_b06 = calculate_ndre_b06(red_edge_2, nir_2)

    print("NDRE-B06 calculated successfully!")
    print("\nNDRE-B06 statistics:")
    print("Minimum:", np.nanmin(ndre_b06))
    print("Maximum:", np.nanmax(ndre_b06))
    print("Mean:", np.nanmean(ndre_b06))
    print("Median:", np.nanmedian(ndre_b06))
    print("Standard deviation:", np.nanstd(ndre_b06))