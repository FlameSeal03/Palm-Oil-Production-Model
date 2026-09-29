"""
Palm Canopy Mask

This module provides two ways to identify palm-canopy candidate pixels:

1. THRESHOLD MODE
   Uses fixed spectral thresholds for:
       - NDVI
       - NDRE
       - NDRE-B06
       - NDRE-B07

2. CONTINUOUS MODE
   Produces a continuous 0-1 Palm Canopy Likelihood Score based on
   how strongly each pixel meets the selected spectral thresholds.

Important:
    This is a spectral palm-canopy candidate mask, not a validated
    species classifier. Sentinel-2 pixels are 10 m, so individual
    palm trees/crowns are not being directly identified.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.common.sentinel2_data import (
    red,
    red_edge_1,
    red_edge_2,
    red_edge_3,
    nir,
    nir_2,
    plantation_geometry_utm,
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.ndre_b06 import calculate_ndre_b06
from src.indices.ndre_b07 import calculate_ndre_b07


# ============================================================
# USER SETTINGS
# ============================================================

# Choose how the palm mask is calculated:
#
# "threshold"  = binary mask using fixed thresholds
# "continuous" = continuous 0-1 likelihood score
#

MASK_MODE = "threshold"

# MASK_MODE = "continuous"


# ------------------------------------------------------------
# Threshold sets
# ------------------------------------------------------------

# Broad:
# More inclusive. Captures more potential palm-canopy pixels.
BROAD_THRESHOLDS = {
    "ndvi": 0.60,
    "ndre": 0.40,
    "ndre_b06": 0.10,
    "ndre_b07": 0.02,
}


# Balanced:
# Middle-ground option between broad and conservative.
BALANCED_THRESHOLDS = {
    "ndvi": 0.65,
    "ndre": 0.45,
    "ndre_b06": 0.13,
    "ndre_b07": 0.039,
}


# Conservative:
# More selective. Only stronger spectral candidates are retained.
CONSERVATIVE_THRESHOLDS = {
    "ndvi": 0.75,
    "ndre": 0.52,
    "ndre_b06": 0.145,
    "ndre_b07": 0.047,
}


# Which threshold set should be used?
#
# Options:
#     "broad"
#     "balanced"
#     "conservative"
#
THRESHOLD_PRESET = "balanced"


# ============================================================
# LOAD / CALCULATE FEATURES
# ============================================================

ndvi = calculate_ndvi(nir, red)
ndre = calculate_ndre(nir_2, red_edge_1)
ndre_b06 = calculate_ndre_b06(nir_2, red_edge_2)
ndre_b07 = calculate_ndre_b07(nir_2, red_edge_3)


# ============================================================
# SELECT THRESHOLDS
# ============================================================

THRESHOLD_SETS = {
    "broad": BROAD_THRESHOLDS,
    "balanced": BALANCED_THRESHOLDS,
    "conservative": CONSERVATIVE_THRESHOLDS,
}

if THRESHOLD_PRESET not in THRESHOLD_SETS:
    raise ValueError(
        f"Invalid THRESHOLD_PRESET: {THRESHOLD_PRESET}. "
        f"Choose from: {list(THRESHOLD_SETS.keys())}"
    )

thresholds = THRESHOLD_SETS[THRESHOLD_PRESET]


# ============================================================
# VALID PIXELS
# ============================================================

valid_pixels = (
    np.isfinite(ndvi)
    & np.isfinite(ndre)
    & np.isfinite(ndre_b06)
    & np.isfinite(ndre_b07)
)


# ============================================================
# THRESHOLD MASK
# ============================================================

ndvi_pass = ndvi >= thresholds["ndvi"]
ndre_pass = ndre >= thresholds["ndre"]
ndre_b06_pass = ndre_b06 >= thresholds["ndre_b06"]
ndre_b07_pass = ndre_b07 >= thresholds["ndre_b07"]


threshold_mask = (
    valid_pixels
    & ndvi_pass
    & ndre_pass
    & ndre_b06_pass
    & ndre_b07_pass
)


# ============================================================
# CONTINUOUS PALM CANOPY LIKELIHOOD
# ============================================================

def calculate_continuous_score(
    values,
    broad_threshold,
    conservative_threshold,
):
    """
    Convert a spectral feature into a 0-1 score.

    0:
        At or below the broad threshold.

    1:
        At or above the conservative threshold.

    Between 0 and 1:
        Linearly scaled between the two thresholds.

    Values above the conservative threshold are capped at 1.
    Values below the broad threshold are capped at 0.
    """

    denominator = conservative_threshold - broad_threshold

    if denominator <= 0:
        raise ValueError(
            "Conservative threshold must be greater than "
            "the broad threshold."
        )

    score = (
        (values - broad_threshold)
        / denominator
    )

    return np.clip(score, 0.0, 1.0)


ndvi_score = calculate_continuous_score(
    ndvi,
    BROAD_THRESHOLDS["ndvi"],
    CONSERVATIVE_THRESHOLDS["ndvi"],
)

ndre_score = calculate_continuous_score(
    ndre,
    BROAD_THRESHOLDS["ndre"],
    CONSERVATIVE_THRESHOLDS["ndre"],
)

ndre_b06_score = calculate_continuous_score(
    ndre_b06,
    BROAD_THRESHOLDS["ndre_b06"],
    CONSERVATIVE_THRESHOLDS["ndre_b06"],
)

ndre_b07_score = calculate_continuous_score(
    ndre_b07,
    BROAD_THRESHOLDS["ndre_b07"],
    CONSERVATIVE_THRESHOLDS["ndre_b07"],
)


continuous_score = (
    ndvi_score
    + ndre_score
    + ndre_b06_score
    + ndre_b07_score
) / 4.0


# Make invalid pixels NaN rather than treating them as zero.
continuous_score[~valid_pixels] = np.nan


# ============================================================
# CHOOSE ACTIVE OUTPUT
# ============================================================

if MASK_MODE == "threshold":

    palm_mask = threshold_mask

elif MASK_MODE == "continuous":

    palm_mask = continuous_score

else:

    raise ValueError(
        f"Invalid MASK_MODE: {MASK_MODE}. "
        "Choose either 'threshold' or 'continuous'."
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

valid_count = np.sum(valid_pixels)

print()
print("=" * 60)
print("PALM CANOPY MASK")
print("=" * 60)

print(f"Mode: {MASK_MODE}")
print(f"Threshold preset: {THRESHOLD_PRESET}")

print()
print("Thresholds:")
print(f"  NDVI:      {thresholds['ndvi']}")
print(f"  NDRE:      {thresholds['ndre']}")
print(f"  NDRE-B06:  {thresholds['ndre_b06']}")
print(f"  NDRE-B07:  {thresholds['ndre_b07']}")


# ------------------------------------------------------------
# Individual threshold diagnostics
# ------------------------------------------------------------

print()
print("Individual threshold results:")

print(
    f"NDVI >= {thresholds['ndvi']}: "
    f"{np.sum(valid_pixels & ndvi_pass):,} "
    f"({np.sum(valid_pixels & ndvi_pass) / valid_count * 100:.2f}%)"
)

print(
    f"NDRE >= {thresholds['ndre']}: "
    f"{np.sum(valid_pixels & ndre_pass):,} "
    f"({np.sum(valid_pixels & ndre_pass) / valid_count * 100:.2f}%)"
)

print(
    f"NDRE-B06 >= {thresholds['ndre_b06']}: "
    f"{np.sum(valid_pixels & ndre_b06_pass):,} "
    f"({np.sum(valid_pixels & ndre_b06_pass) / valid_count * 100:.2f}%)"
)

print(
    f"NDRE-B07 >= {thresholds['ndre_b07']}: "
    f"{np.sum(valid_pixels & ndre_b07_pass):,} "
    f"({np.sum(valid_pixels & ndre_b07_pass) / valid_count * 100:.2f}%)"
)


# ------------------------------------------------------------
# Cumulative threshold diagnostics
# ------------------------------------------------------------

cumulative_ndvi = valid_pixels & ndvi_pass

cumulative_ndre = (
    cumulative_ndvi
    & ndre_pass
)

cumulative_b06 = (
    cumulative_ndre
    & ndre_b06_pass
)

cumulative_b07 = (
    cumulative_b06
    & ndre_b07_pass
)

print()
print("Cumulative threshold results:")

print(
    f"After NDVI:      "
    f"{np.sum(cumulative_ndvi):,} "
    f"({np.sum(cumulative_ndvi) / valid_count * 100:.2f}%)"
)

print(
    f"After NDRE:      "
    f"{np.sum(cumulative_ndre):,} "
    f"({np.sum(cumulative_ndre) / valid_count * 100:.2f}%)"
)

print(
    f"After NDRE-B06:  "
    f"{np.sum(cumulative_b06):,} "
    f"({np.sum(cumulative_b06) / valid_count * 100:.2f}%)"
)

print(
    f"After NDRE-B07:  "
    f"{np.sum(cumulative_b07):,} "
    f"({np.sum(cumulative_b07) / valid_count * 100:.2f}%)"
)


# ============================================================
# THRESHOLD MODE RESULTS
# ============================================================

threshold_count = np.sum(threshold_mask)

print()
print("Threshold mask:")
print(
    f"Palm-canopy candidate pixels: "
    f"{threshold_count:,} / {valid_count:,}"
)

print(
    f"Percentage of valid pixels: "
    f"{threshold_count / valid_count * 100:.2f}%"
)


# ============================================================
# CONTINUOUS MODE RESULTS
# ============================================================

valid_scores = continuous_score[valid_pixels]

print()
print("Continuous Palm Canopy Likelihood Score:")

print(
    f"Minimum:  {np.nanmin(valid_scores):.4f}"
)

print(
    f"Maximum:  {np.nanmax(valid_scores):.4f}"
)

print(
    f"Mean:     {np.nanmean(valid_scores):.4f}"
)

print(
    f"Median:   {np.nanmedian(valid_scores):.4f}"
)

print(
    f"Std dev:  {np.nanstd(valid_scores):.4f}"
)


for score_threshold in [0.25, 0.50, 0.75, 0.90]:

    count = np.sum(
        valid_pixels
        & (continuous_score >= score_threshold)
    )

    percentage = count / valid_count * 100

    print(
        f"Score >= {score_threshold:.2f}: "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# FINAL OUTPUT STATISTICS
# ============================================================

if MASK_MODE == "threshold":

    final_count = np.sum(palm_mask)

    print()
    print("ACTIVE MASK")
    print("-" * 60)
    print("Binary threshold mask selected.")
    print(
        f"Candidate pixels: {final_count:,} "
        f"({final_count / valid_count * 100:.2f}%)"
    )

else:

    print()
    print("ACTIVE MASK")
    print("-" * 60)
    print("Continuous 0-1 Palm Canopy Likelihood selected.")


print("=" * 60)
print()


# ============================================================
# VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 10),
)


# ------------------------------------------------------------
# NDVI
# ------------------------------------------------------------

im1 = axes[0, 0].imshow(
    np.where(valid_pixels, ndvi, np.nan),
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
)

axes[0, 0].set_title("NDVI")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)


# ------------------------------------------------------------
# NDRE
# ------------------------------------------------------------

im2 = axes[0, 1].imshow(
    np.where(valid_pixels, ndre, np.nan),
    cmap="viridis",
)

axes[0, 1].set_title("NDRE")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)


# ------------------------------------------------------------
# NDRE-B06
# ------------------------------------------------------------

im3 = axes[0, 2].imshow(
    np.where(valid_pixels, ndre_b06, np.nan),
    cmap="viridis",
)

axes[0, 2].set_title("NDRE-B06")
axes[0, 2].axis("off")
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)


# ------------------------------------------------------------
# NDRE-B07
# ------------------------------------------------------------

im4 = axes[1, 0].imshow(
    np.where(valid_pixels, ndre_b07, np.nan),
    cmap="viridis",
)

axes[1, 0].set_title("NDRE-B07")
axes[1, 0].axis("off")
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)


# ------------------------------------------------------------
# Threshold mask
# ------------------------------------------------------------

axes[1, 1].imshow(
    threshold_mask,
    cmap="gray",
    vmin=0,
    vmax=1,
)

axes[1, 1].set_title(
    f"Threshold Mask ({THRESHOLD_PRESET})"
)

axes[1, 1].axis("off")


# ------------------------------------------------------------
# Continuous score
# ------------------------------------------------------------

im6 = axes[1, 2].imshow(
    continuous_score,
    cmap="viridis",
    vmin=0,
    vmax=1,
)

axes[1, 2].set_title(
    "Continuous Palm Canopy Score"
)

axes[1, 2].axis("off")

plt.colorbar(
    im6,
    ax=axes[1, 2],
    fraction=0.046,
)


plt.suptitle(
    f"Palm Canopy Mask Analysis — {MASK_MODE.upper()} MODE",
    fontsize=16,
)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

output_path = (
    "outputs/"
    f"palm_mask_{MASK_MODE}_{THRESHOLD_PRESET}.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Visualization saved to: {output_path}")