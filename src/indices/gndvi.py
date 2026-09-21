from src.common.sentinel2_data import green, nir
import numpy as np

denominator = nir + green

gndvi = np.where(
    denominator != 0,
    (nir - green) / denominator,
    np.nan
)

print("GNDVI calculated successfully!")

print("\nGNDVI statistics:")
print("Minimum:", np.nanmin(gndvi))
print("Maximum:", np.nanmax(gndvi))
print("Mean:", np.nanmean(gndvi))
print("Median:", np.nanmedian(gndvi))
print("Standard deviation:", np.nanstd(gndvi))