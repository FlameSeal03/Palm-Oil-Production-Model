from src.common.sentinel2_data import red_edge_1, nir_2
import numpy as np

denominator = nir_2 + red_edge_1

ndre = np.where(
    denominator != 0,
    (nir_2 - red_edge_1) / denominator,
    np.nan
)

print("NDRE calculated successfully!")

print("\nNDRE statistics:")
print("Minimum:", np.nanmin(ndre))
print("Maximum:", np.nanmax(ndre))
print("Mean:", np.nanmean(ndre))
print("Median:", np.nanmedian(ndre))
print("Standard deviation:", np.nanstd(ndre))