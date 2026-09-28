import numpy as np


def calculate_ndre_b07(red_edge_3, nir_2):
    """
    Calculate an NDRE-style index using Sentinel-2 B07 and B8A.

    NDRE-B07 = (B8A - B07) / (B8A + B07)
    """

    denominator = nir_2 + red_edge_3

    ndre_b07 = np.where(
        denominator != 0,
        (nir_2 - red_edge_3) / denominator,
        np.nan
    )

    return ndre_b07


if __name__ == "__main__":
    from src.common.sentinel2_data import red_edge_3, nir_2

    ndre_b07 = calculate_ndre_b07(red_edge_3, nir_2)

    print("NDRE-B07 calculated successfully!")
    print("\nNDRE-B07 statistics:")
    print("Minimum:", np.nanmin(ndre_b07))
    print("Maximum:", np.nanmax(ndre_b07))
    print("Mean:", np.nanmean(ndre_b07))
    print("Median:", np.nanmedian(ndre_b07))
    print("Standard deviation:", np.nanstd(ndre_b07))