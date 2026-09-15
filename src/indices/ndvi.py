from src.common.sentinel2_data import red, nir

import numpy as np


# NDVI
denominator = nir + red

ndvi = np.where(
    denominator != 0,
    (nir - red) / denominator,
    np.nan
)


print("NDVI calculated successfully!")

print("\nNDVI statistics:")
print("Minimum:", np.nanmin(ndvi))
print("Maximum:", np.nanmax(ndvi))
print("Mean:", np.nanmean(ndvi))
print("Median:", np.nanmedian(ndvi))
print("Standard deviation:", np.nanstd(ndvi))