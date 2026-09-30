"""
Palm Canopy Mask

Provides two palm-canopy masking approaches:

1. THRESHOLD MODE
   Uses fixed spectral thresholds to create a binary mask.

2. CONTINUOUS MODE
   Creates a continuous 0-1 spectral score.

The main compatibility function is:

    calculate_palm_mask()

This allows other modules, such as plantation_image.py, to use the
palm mask without needing to know which mode is currently selected.

Important:
    This is a spectral palm-canopy candidate mask, not a validated
    species classifier. Sentinel-2 pixels are 10 m, so individual
    palm trees/crowns are not directly identified.
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
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.ndre_b06 import calculate_ndre_b06
from src.indices.ndre_b07 import calculate_ndre_b07


# ============================================================
# USER SETTINGS
# ============================================================

# Choose the active palm-mask method:
#
# "threshold"  = binary mask
# "continuous" = 0-1 continuous score
#
MASK_MODE = "continuous"


# ============================================================
# THRESHOLD SETTINGS
# ============================================================

# Broad:
# More inclusive. Captures more potential palm-canopy pixels.
BROAD_THRESHOLDS = {
    "ndvi": 0.60,
    "ndre": 0.40,
    "ndre_b06": 0.10,
    "ndre_b07": 0.02,
}


# Balanced:
# Middle-ground option.
BALANCED_THRESHOLDS = {
    "ndvi": 0.65,
    "ndre": 0.45,
    "ndre_b06": 0.13,
    "ndre_b07": 0.039,
}


# Conservative:
# More selective. Keeps stronger spectral candidates.
CONSERVATIVE_THRESHOLDS = {
    "ndvi": 0.75,
    "ndre": 0.52,
    "ndre_b06": 0.145,
    "ndre_b07": 0.047,
}


THRESHOLD_SETS = {
    "broad": BROAD_THRESHOLDS,
    "balanced": BALANCED_THRESHOLDS,
    "conservative": CONSERVATIVE_THRESHOLDS,
}


# Select which threshold set to use.
#
# Options:
#     "broad"
#     "balanced"
#     "conservative"
#
THRESHOLD_PRESET = "balanced"


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features():
    """
    Calculate the four spectral features used by the palm mask.

    Returns:
        tuple:
            ndvi
            ndre
            ndre_b06
            ndre_b07
    """

    ndvi = calculate_ndvi(nir, red)

    ndre = calculate_ndre(
        nir_2,
        red_edge_1,
    )

    ndre_b06 = calculate_ndre_b06(
        nir_2,
        red_edge_2,
    )

    ndre_b07 = calculate_ndre_b07(
        nir_2,
        red_edge_3,
    )

    return (
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )


# ============================================================
# VALID PIXEL MASK
# ============================================================

def calculate_valid_pixels(
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
):
    """
    Identify pixels where all four spectral features are valid.
    """

    return (
        np.isfinite(ndvi)
        & np.isfinite(ndre)
        & np.isfinite(ndre_b06)
        & np.isfinite(ndre_b07)
    )


# ============================================================
# THRESHOLD MASK
# ============================================================

def calculate_threshold_mask(
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
    thresholds,
):
    """
    Create a binary palm-canopy candidate mask.

    A pixel must pass all four spectral thresholds.
    """

    valid_pixels = calculate_valid_pixels(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

    mask = (
        valid_pixels
        & (ndvi >= thresholds["ndvi"])
        & (ndre >= thresholds["ndre"])
        & (ndre_b06 >= thresholds["ndre_b06"])
        & (ndre_b07 >= thresholds["ndre_b07"])
    )

    return mask


# ============================================================
# CONTINUOUS SCORE
# ============================================================

def calculate_continuous_score(
    values,
    broad_threshold,
    conservative_threshold,
):
    """
    Convert a feature into a 0-1 continuous score.

    0:
        At or below the broad threshold.

    1:
        At or above the conservative threshold.

    Between 0 and 1:
        Linearly scaled between the two thresholds.
    """

    denominator = (
        conservative_threshold
        - broad_threshold
    )

    if denominator <= 0:
        raise ValueError(
            "Conservative threshold must be greater "
            "than the broad threshold."
        )

    score = (
        (values - broad_threshold)
        / denominator
    )

    return np.clip(
        score,
        0.0,
        1.0,
    )


def calculate_palm_likelihood(
    ndvi,
    ndre,
    ndre_b06,
    ndre_b07,
):
    """
    Calculate the continuous 0-1 Palm Canopy Likelihood Score.

    This is a relative spectral score, NOT a calibrated probability.
    """

    valid_pixels = calculate_valid_pixels(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

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

    likelihood = (
        ndvi_score
        + ndre_score
        + ndre_b06_score
        + ndre_b07_score
    ) / 4.0

    likelihood[~valid_pixels] = np.nan

    return likelihood


# ============================================================
# MAIN COMPATIBILITY FUNCTION
# ============================================================

def calculate_palm_mask(
    ndvi=None,
    ndre=None,
    ndre_b06=None,
    ndre_b07=None,
    ndvi_threshold=None,
    ndre_threshold=None,
    ndre_b06_threshold=None,
    ndre_b07_threshold=None,
):
    """
    Calculate the palm-canopy mask.

    Supports both the new two-mode system and the older function
    interface used by existing visualization code.

    Parameters
    ----------
    ndvi : numpy.ndarray, optional
        NDVI array. If omitted, NDVI is calculated from the
        Sentinel-2 data loaded by this module.

    ndre : numpy.ndarray, optional
        NDRE array. If omitted, NDRE is calculated automatically.

    ndre_b06 : numpy.ndarray, optional
        NDRE using B06. If omitted, it is calculated automatically.

    ndre_b07 : numpy.ndarray, optional
        NDRE using B07. If omitted, it is calculated automatically.

    ndvi_threshold : float, optional
        Backward-compatible NDVI threshold.

    ndre_threshold : float, optional
        Backward-compatible NDRE threshold.

    ndre_b06_threshold : float, optional
        Optional NDRE-B06 threshold.

    ndre_b07_threshold : float, optional
        Optional NDRE-B07 threshold.

    Returns
    -------
    numpy.ndarray
        Threshold mode:
            Boolean mask.

        Continuous mode:
            Floating-point 0-1 likelihood score.
    """

    # --------------------------------------------------------
    # Calculate missing features
    # --------------------------------------------------------

    if ndvi is None:
        ndvi = calculate_ndvi(
            nir,
            red,
        )

    if ndre is None:
        ndre = calculate_ndre(
            nir_2,
            red_edge_1,
        )

    if ndre_b06 is None:
        ndre_b06 = calculate_ndre_b06(
            nir_2,
            red_edge_2,
        )

    if ndre_b07 is None:
        ndre_b07 = calculate_ndre_b07(
            nir_2,
            red_edge_3,
        )

    # --------------------------------------------------------
    # CONTINUOUS MODE
    # --------------------------------------------------------
    #
    # If the user selects continuous mode, use the full
    # four-feature continuous score.
    #
    # The old threshold arguments are intentionally ignored
    # here because continuous mode has its own definition.
    # --------------------------------------------------------

    if MASK_MODE == "continuous":

        return calculate_palm_likelihood(
            ndvi,
            ndre,
            ndre_b06,
            ndre_b07,
        )

    # --------------------------------------------------------
    # THRESHOLD MODE
    # --------------------------------------------------------

    if MASK_MODE != "threshold":

        raise ValueError(
            f"Invalid MASK_MODE: {MASK_MODE}. "
            "Choose either 'threshold' or 'continuous'."
        )

    # --------------------------------------------------------
    # Backward compatibility
    # --------------------------------------------------------
    #
    # If the old visualization code supplies NDVI/NDRE
    # thresholds, use them.
    #
    # Otherwise use the currently selected preset.
    # --------------------------------------------------------

    if (
        ndvi_threshold is not None
        or ndre_threshold is not None
        or ndre_b06_threshold is not None
        or ndre_b07_threshold is not None
    ):

        thresholds = {
            "ndvi": (
                ndvi_threshold
                if ndvi_threshold is not None
                else BALANCED_THRESHOLDS["ndvi"]
            ),

            "ndre": (
                ndre_threshold
                if ndre_threshold is not None
                else BALANCED_THRESHOLDS["ndre"]
            ),

            "ndre_b06": (
                ndre_b06_threshold
                if ndre_b06_threshold is not None
                else BALANCED_THRESHOLDS["ndre_b06"]
            ),

            "ndre_b07": (
                ndre_b07_threshold
                if ndre_b07_threshold is not None
                else BALANCED_THRESHOLDS["ndre_b07"]
            ),
        }

    else:

        if THRESHOLD_PRESET not in THRESHOLD_SETS:

            raise ValueError(
                f"Invalid THRESHOLD_PRESET: "
                f"{THRESHOLD_PRESET}. "
                f"Choose from: "
                f"{list(THRESHOLD_SETS.keys())}"
            )

        thresholds = THRESHOLD_SETS[
            THRESHOLD_PRESET
        ]

    # --------------------------------------------------------
    # Calculate binary mask
    # --------------------------------------------------------

    return calculate_threshold_mask(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
        thresholds,
    )


# ============================================================
# DIAGNOSTIC SCRIPT
# ============================================================

def main():

    (
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    ) = calculate_features()

    valid_pixels = calculate_valid_pixels(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

    valid_count = np.sum(valid_pixels)

    if valid_count == 0:
        raise RuntimeError(
            "No valid pixels were found. "
            "Check the Sentinel-2 imagery and plantation geometry."
        )

    if THRESHOLD_PRESET not in THRESHOLD_SETS:
        raise ValueError(
            f"Invalid THRESHOLD_PRESET: "
            f"{THRESHOLD_PRESET}"
        )

    thresholds = THRESHOLD_SETS[
        THRESHOLD_PRESET
    ]

    # --------------------------------------------------------
    # Individual threshold results
    # --------------------------------------------------------

    ndvi_pass = (
        ndvi >= thresholds["ndvi"]
    )

    ndre_pass = (
        ndre >= thresholds["ndre"]
    )

    ndre_b06_pass = (
        ndre_b06 >= thresholds["ndre_b06"]
    )

    ndre_b07_pass = (
        ndre_b07 >= thresholds["ndre_b07"]
    )

    # --------------------------------------------------------
    # Cumulative masks
    # --------------------------------------------------------

    cumulative_ndvi = (
        valid_pixels
        & ndvi_pass
    )

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

    threshold_mask = cumulative_b07

    # --------------------------------------------------------
    # Continuous score
    # --------------------------------------------------------

    continuous_score = calculate_palm_likelihood(
        ndvi,
        ndre,
        ndre_b06,
        ndre_b07,
    )

    # ========================================================
    # PRINT DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 60)
    print("PALM CANOPY MASK")
    print("=" * 60)

    print(f"Mode: {MASK_MODE}")
    print(f"Threshold preset: {THRESHOLD_PRESET}")

    print()
    print("Thresholds:")
    print(
        f"  NDVI:      {thresholds['ndvi']}"
    )
    print(
        f"  NDRE:      {thresholds['ndre']}"
    )
    print(
        f"  NDRE-B06:  {thresholds['ndre_b06']}"
    )
    print(
        f"  NDRE-B07:  {thresholds['ndre_b07']}"
    )

    # --------------------------------------------------------
    # Individual thresholds
    # --------------------------------------------------------

    print()
    print("Individual threshold results:")

    count = np.sum(
        valid_pixels & ndvi_pass
    )

    print(
        f"NDVI >= {thresholds['ndvi']}: "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(
        valid_pixels & ndre_pass
    )

    print(
        f"NDRE >= {thresholds['ndre']}: "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(
        valid_pixels & ndre_b06_pass
    )

    print(
        f"NDRE-B06 >= {thresholds['ndre_b06']}: "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(
        valid_pixels & ndre_b07_pass
    )

    print(
        f"NDRE-B07 >= {thresholds['ndre_b07']}: "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Cumulative thresholds
    # --------------------------------------------------------

    print()
    print("Cumulative threshold results:")

    count = np.sum(cumulative_ndvi)

    print(
        f"After NDVI:      "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(cumulative_ndre)

    print(
        f"After NDRE:      "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(cumulative_b06)

    print(
        f"After NDRE-B06:  "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    count = np.sum(cumulative_b07)

    print(
        f"After NDRE-B07:  "
        f"{count:,} "
        f"({count / valid_count * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Binary mask
    # --------------------------------------------------------

    threshold_count = np.sum(
        threshold_mask
    )

    print()
    print("Threshold mask:")
    print(
        f"Palm-canopy candidate pixels: "
        f"{threshold_count:,} / "
        f"{valid_count:,}"
    )

    print(
        f"Percentage of valid pixels: "
        f"{threshold_count / valid_count * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Continuous score
    # --------------------------------------------------------

    valid_scores = (
        continuous_score[valid_pixels]
    )

    print()
    print(
        "Continuous Palm Canopy "
        "Likelihood Score:"
    )

    print(
        f"Minimum:  "
        f"{np.nanmin(valid_scores):.4f}"
    )

    print(
        f"Maximum:  "
        f"{np.nanmax(valid_scores):.4f}"
    )

    print(
        f"Mean:     "
        f"{np.nanmean(valid_scores):.4f}"
    )

    print(
        f"Median:   "
        f"{np.nanmedian(valid_scores):.4f}"
    )

    print(
        f"Std dev:  "
        f"{np.nanstd(valid_scores):.4f}"
    )

    for score_threshold in [
        0.25,
        0.50,
        0.75,
        0.90,
    ]:

        count = np.sum(
            valid_pixels
            & (
                continuous_score
                >= score_threshold
            )
        )

        percentage = (
            count
            / valid_count
            * 100
        )

        print(
            f"Score >= "
            f"{score_threshold:.2f}: "
            f"{count:,} pixels "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # ACTIVE MASK
    # ========================================================

    print()
    print("ACTIVE MASK")
    print("-" * 60)

    if MASK_MODE == "threshold":

        print(
            "Binary threshold mask selected."
        )

        print(
            f"Candidate pixels: "
            f"{threshold_count:,} "
            f"({threshold_count / valid_count * 100:.2f}%)"
        )

    else:

        print(
            "Continuous 0-1 Palm Canopy "
            "Likelihood selected."
        )

    print("=" * 60)
    print()

    # ========================================================
    # VISUALIZATION
    # ========================================================

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    im1 = axes[0, 0].imshow(
        np.where(
            valid_pixels,
            ndvi,
            np.nan,
        ),
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )

    axes[0, 0].set_title("NDVI")
    axes[0, 0].axis("off")

    plt.colorbar(
        im1,
        ax=axes[0, 0],
        fraction=0.046,
    )

    # --------------------------------------------------------
    # NDRE
    # --------------------------------------------------------

    im2 = axes[0, 1].imshow(
        np.where(
            valid_pixels,
            ndre,
            np.nan,
        ),
        cmap="viridis",
    )

    axes[0, 1].set_title("NDRE")
    axes[0, 1].axis("off")

    plt.colorbar(
        im2,
        ax=axes[0, 1],
        fraction=0.046,
    )

    # --------------------------------------------------------
    # NDRE-B06
    # --------------------------------------------------------

    im3 = axes[0, 2].imshow(
        np.where(
            valid_pixels,
            ndre_b06,
            np.nan,
        ),
        cmap="viridis",
    )

    axes[0, 2].set_title("NDRE-B06")
    axes[0, 2].axis("off")

    plt.colorbar(
        im3,
        ax=axes[0, 2],
        fraction=0.046,
    )

    # --------------------------------------------------------
    # NDRE-B07
    # --------------------------------------------------------

    im4 = axes[1, 0].imshow(
        np.where(
            valid_pixels,
            ndre_b07,
            np.nan,
        ),
        cmap="viridis",
    )

    axes[1, 0].set_title("NDRE-B07")
    axes[1, 0].axis("off")

    plt.colorbar(
        im4,
        ax=axes[1, 0],
        fraction=0.046,
    )

    # --------------------------------------------------------
    # Threshold mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Continuous score
    # --------------------------------------------------------

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
        f"Palm Canopy Mask Analysis — "
        f"{MASK_MODE.upper()} MODE",
        fontsize=16,
    )

    plt.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        "outputs/"
        f"palm_mask_"
        f"{MASK_MODE}_"
        f"{THRESHOLD_PRESET}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(
        f"Visualization saved to: "
        f"{output_path}"
    )


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()