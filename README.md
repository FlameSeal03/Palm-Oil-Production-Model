# Palm-Oil-Production-Model

A Python-based satellite imagery pipeline for extracting metrics from oil-palm plantations. The long-term goal is to use satellite-derived features to build an oil-palm production model.

## Current Capabilities

The project currently uses **Sentinel-2 Level-2A imagery** from Copernicus Data Space to:

* Load plantation boundaries from GeoJSON files.
* Search for Sentinel-2 imagery over a plantation and date range.
* Filter scenes by cloud cover.
* Retrieve satellite bands at approximately 10 m resolution.
* Apply Sentinel-2 Scene Classification Layer (SCL) filtering.
* Mask imagery to the exact plantation boundary.
* Calculate vegetation and moisture indices.
* Create monthly composites from multiple Sentinel-2 scenes.
* Track the number of valid observations for each pixel.
* Log monthly composite results to a text file.

## Current Indices

| Index     | Purpose                              |
| --------- | ------------------------------------ |
| **NDVI**  | General vegetation vigor             |
| **NDRE**  | Vegetation/chlorophyll signal        |
| **GNDVI** | Vegetation and chlorophyll signal    |
| **NDMI**  | Vegetation moisture                  |
| **EVI**   | Vegetation vigor in dense vegetation |


The SCL mask currently keeps pixels classified as **vegetation (4)** or **not-vegetated (5)** and removes clouds, cloud shadows, water, and other unusable pixels.

## Running the Project

The config.settings.py file contains location, single day, and data range variables that can be changed for certain planations and time frames. To see what name to include as the location variable, look inside the config.plantation_names.txt to see the planations and their nicknames.

Run an individual index:

```bash
python -m src.indices.ndvi
```

Run all current indices:

```bash
bash run_all_indexes.txt
```

Run the monthly Sentinel-2 composite:

```bash
python -m src.composites.monthly_sentinel2_composite
```

The monthly composite output is displayed in the terminal and saved to:

```text
logs/monthly_sentinel2_composite.log
```

The log file is overwritten with each new run.

## Current Status

**Working prototype.**

The Sentinel-2 search, image retrieval, plantation masking, SCL filtering, five vegetation indices, and monthly composite workflow have been successfully tested.

## Next Steps

* Save monthly composite results as structured data.
* Create a monthly plantation/block feature dataset.
* Improve satellite image quality filtering.
* Add additional satellite-derived features.
* Integrate actual oil-palm production data.
* Develop and evaluate the production model.
