import numpy as np


def calculate_ndre(red_edge_1, nir_2):
    """
    Calculate NDRE from the red-edge and NIR bands.

    NDRE = (NIR - Red Edge) / (NIR + Red Edge)
    """

    denominator = nir_2 + red_edge_1

    ndre = np.where(
        denominator != 0,
        (nir_2 - red_edge_1) / denominator,
        np.nan
    )

    return ndre


if __name__ == "__main__":
    from src.common.sentinel2_data import red_edge_1, nir_2

    ndre = calculate_ndre(red_edge_1, nir_2)

    print("NDRE calculated successfully!")

    print("\nNDRE statistics:")
    print("Minimum:", np.nanmin(ndre))
    print("Maximum:", np.nanmax(ndre))
    print("Mean:", np.nanmean(ndre))
    print("Median:", np.nanmedian(ndre))
    print("Standard deviation:", np.nanstd(ndre))