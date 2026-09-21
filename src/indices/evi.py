from src.common.sentinel2_data import blue, red, nir
import numpy as np

denominator = nir + 6 * red - 7.5 * blue + 1

evi = np.where(
    denominator != 0,
    2.5 * (nir - red) / denominator,
    np.nan
)

print("EVI calculated successfully!")

print("\nEVI statistics:")
print("Minimum:", np.nanmin(evi))
print("Maximum:", np.nanmax(evi))
print("Mean:", np.nanmean(evi))
print("Median:", np.nanmedian(evi))
print("Standard deviation:", np.nanstd(evi))