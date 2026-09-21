from src.common.sentinel2_data import nir, swir_1
import numpy as np

denominator = nir + swir_1

ndmi = np.where(
    denominator != 0,
    (nir - swir_1) / denominator,
    np.nan
)

print("NDMI calculated successfully!")

print("\nNDMI statistics:")
print("Minimum:", np.nanmin(ndmi))
print("Maximum:", np.nanmax(ndmi))
print("Mean:", np.nanmean(ndmi))
print("Median:", np.nanmedian(ndmi))
print("Standard deviation:", np.nanstd(ndmi))