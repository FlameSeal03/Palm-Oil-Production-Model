import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.common.sentinel2_data import (
    red,
    green,
    blue,
    red_edge_1,
    red_edge_2,
    red_edge_3,
    nir,
    nir_2,
    swir_1,
    swir_2,
)

from src.indices.ndvi import calculate_ndvi
from src.indices.ndre import calculate_ndre
from src.indices.gndvi import calculate_gndvi
from src.indices.ndmi import calculate_ndmi
from src.indices.evi import calculate_evi
from src.indices.savi import calculate_savi
from src.indices.msavi import calculate_msavi
from src.indices.nbr import calculate_nbr
from src.indices.ndre_b06 import calculate_ndre_b06
from src.indices.ndre_b07 import calculate_ndre_b07


OUTPUT_PATH = "outputs/feature_correlation.png"


def calculate_features():
    """
    Calculate all satellite-derived features used by the project.

    Returns:
        Dictionary containing each feature as a NumPy array.
    """

    features = {
        "NDVI": calculate_ndvi(red, nir),

        "NDRE": calculate_ndre(
            red_edge_1,
            nir_2,
        ),

        "GNDVI": calculate_gndvi(
            green,
            nir,
        ),

        "NDMI": calculate_ndmi(
            nir,
            swir_1,
        ),

        "EVI": calculate_evi(
            blue,
            red,
            nir,
        ),

        "SAVI": calculate_savi(
            red,
            nir,
        ),

        "MSAVI": calculate_msavi(
            red,
            nir,
        ),

        "NBR": calculate_nbr(
            nir,
            swir_2,
        ),

        "NDRE-B06": calculate_ndre_b06(
            red_edge_2,
            nir_2,
        ),

        "NDRE-B07": calculate_ndre_b07(
            red_edge_3,
            nir_2,
        ),
    }

    return features


def create_dataframe(features):
    """
    Convert the feature arrays into a DataFrame.

    Only pixels where every feature has a finite value
    are retained for the correlation analysis.
    """

    data = {}

    for name, values in features.items():
        data[name] = values.flatten()

    dataframe = pd.DataFrame(data)

    # Remove pixels containing NaN or infinite values.
    dataframe = dataframe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    dataframe = dataframe.dropna()

    return dataframe


def calculate_correlation_matrix(dataframe):
    """
    Calculate the Pearson correlation matrix.
    """

    return dataframe.corr(method="pearson")


def save_correlation_heatmap(correlation_matrix):
    """
    Save the correlation matrix as a heatmap.
    """

    figure, axis = plt.subplots(
        figsize=(12, 10)
    )

    image = axis.imshow(
        correlation_matrix,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_xticks(
        range(len(correlation_matrix.columns))
    )

    axis.set_yticks(
        range(len(correlation_matrix.index))
    )

    axis.set_xticklabels(
        correlation_matrix.columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        correlation_matrix.index,
    )

    # Add correlation values to each cell.
    for row in range(len(correlation_matrix)):
        for column in range(len(correlation_matrix.columns)):
            value = correlation_matrix.iloc[
                row,
                column,
            ]

            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    axis.set_title(
        "Sentinel-2 Feature Correlation"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation",
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def print_correlation_pairs(correlation_matrix):
    """
    Print feature pairs ordered by absolute correlation.

    This helps identify highly redundant features.
    """

    pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            feature_a = columns[i]
            feature_b = columns[j]

            correlation = correlation_matrix.iloc[
                i,
                j,
            ]

            pairs.append(
                (
                    feature_a,
                    feature_b,
                    correlation,
                )
            )

    pairs.sort(
        key=lambda pair: abs(pair[2]),
        reverse=True,
    )

    print("\nFeature pairs ordered by absolute correlation:")
    print("-" * 65)

    for feature_a, feature_b, correlation in pairs:

        print(
            f"{feature_a:12s} <-> "
            f"{feature_b:12s}: "
            f"{correlation: .4f}"
        )


def main():

    print(
        "Calculating Sentinel-2 feature correlations..."
    )

    features = calculate_features()

    dataframe = create_dataframe(
        features
    )

    print(
        f"\nValid pixels used in correlation analysis: "
        f"{len(dataframe)}"
    )

    correlation_matrix = calculate_correlation_matrix(
        dataframe
    )

    print("\nCorrelation matrix:")
    print(
        correlation_matrix.round(4)
    )

    print_correlation_pairs(
        correlation_matrix
    )

    save_correlation_heatmap(
        correlation_matrix
    )

    print(
        f"\nCorrelation heatmap saved to:"
        f"\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()