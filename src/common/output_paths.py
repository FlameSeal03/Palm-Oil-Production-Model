from pathlib import Path

from config.settings import GEOJSON_PATH


def get_plantation_name():
    """
    Get a short plantation name from the GeoJSON filename.

    Examples:
        ANTANG.geojson
            -> ANTANG

        PT. AGRO GADING SEJAHTERA.geojson
            -> AGRO

        PT.geojson
            -> PT
    """

    filename = Path(GEOJSON_PATH).stem

    if filename.upper().startswith("PT. "):
        filename = filename[4:]

    # Use the first word as the short plantation name.
    plantation_name = filename.split()[0]

    return plantation_name.upper()


def get_output_directory():
    """
    Return the output directory for the currently selected plantation.
    Create it if it does not already exist.
    """

    plantation_name = get_plantation_name()

    output_directory = (
        Path("outputs") / plantation_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory