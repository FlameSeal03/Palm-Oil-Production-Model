# Palm-Oil-Production-Model

A Python-based satellite imagery pipeline for extracting features from oil-palm plantation blocks. The long-term goal is to use satellite-derived features to develop and evaluate an oil-palm production model.

The project currently focuses on **Sentinel-2 satellite imagery**, plantation-level geospatial processing, vegetation indices, palm vegetation masking, and monthly satellite composites.

---

## Project Goal

The goal of this project is to build a production modeling pipeline where satellite imagery is used to generate measurable features for oil-palm plantation blocks.

The general workflow is:

```text
Plantation Boundary
        ↓
Sentinel-2 Satellite Imagery
        ↓
Image Quality / Cloud Filtering
        ↓
Plantation Boundary Mask
        ↓
Vegetation & Moisture Indices
        ↓
Palm Vegetation Mask
        ↓
Monthly / Time-Series Features
        ↓
Production Data
        ↓
Oil-Palm Production Model
```

The production data will be added later and used as the target variable for modeling.

---

# Current Capabilities

The project currently uses **Sentinel-2 Level-2A imagery from Copernicus Data Space** to:

* Load plantation boundaries from GeoJSON files.
* Search Sentinel-2 imagery for a plantation and date range.
* Identify individual Sentinel-2 scenes by cloud cover.
* Retrieve Sentinel-2 bands through the Copernicus Data Space Process API.
* Work with Sentinel-2 imagery at approximately 10 m resolution.
* Reproject plantation boundaries into the imagery coordinate system.
* Crop imagery to the plantation boundary.
* Apply Sentinel-2 Scene Classification Layer (SCL) filtering.
* Calculate multiple vegetation and moisture indices.
* Create a simple palm vegetation mask.
* Compare vegetation features and their correlations.
* Create plantation visualization images.
* Create monthly Sentinel-2 composites from multiple scenes.
* Track the number of valid satellite observations for each pixel.
* Save monthly composite processing information to log files.

---

# Satellite Data

## Sentinel-2

The current project uses **Sentinel-2 Level-2A** imagery from Copernicus Data Space.

Sentinel-2 provides multispectral imagery that can be used to measure vegetation characteristics such as:

* General vegetation vigor
* Chlorophyll-related signals
* Vegetation moisture
* Dense vegetation response
* Vegetation stress
* Burn or disturbance-related signals

The project uses multiple Sentinel-2 bands, including:

* Blue
* Green
* Red
* Red Edge
* Near Infrared (NIR)
* Short-Wave Infrared (SWIR)

---

# Plantation Boundaries

Plantation boundaries are stored as GeoJSON files in:

```text
data/boundaries/
```

Current plantation boundary files include:

```text
PT.geojson
PT. AGRO GADING SEJAHTERA.geojson
ANTANG.geojson
```

The selected plantation is controlled through:

```text
config/settings.py
```

For example:

```python
GEOJSON_PATH = "data/boundaries/ANTANG.geojson"
```

This allows the same processing pipeline to be used for different plantation blocks without changing the underlying processing code.

---

# Sentinel-2 Scene Search

The project contains tools for finding Sentinel-2 imagery over a plantation.

## Best Individual Scene

The best available scene can be searched using:

```bash
python -m src.search.best_sentinel2_date
```

The search:

1. Loads the selected plantation boundary.
2. Searches Copernicus Data Space.
3. Finds Sentinel-2 Level-2A scenes covering the plantation.
4. Retrieves cloud-cover information.
5. Sorts scenes by cloud cover.
6. Displays the best available dates.

For example, one test of the `ANTANG` plantation found:

```text
2026-07-20
Cloud cover: 0.02%
```

This allows the project to select a relatively clear image for single-date analysis.

---

# Monthly Sentinel-2 Search

The project also contains a monthly search workflow:

```bash
python -m src.search.monthly_sentinel2_search
```

This examines Sentinel-2 scenes across a date range and organizes available imagery by month.

The monthly search is useful for determining:

* How many images are available each month
* Cloud coverage
* Which months have sufficient imagery
* Which months may require additional filtering or compositing

---

# Vegetation and Moisture Indices

The project currently calculates multiple Sentinel-2 indices.

| Index    | Main Purpose                                        |
| -------- | --------------------------------------------------- |
| NDVI     | General vegetation vigor                            |
| NDRE     | Chlorophyll / vegetation condition                  |
| GNDVI    | Chlorophyll and vegetation response                 |
| NDMI     | Vegetation moisture                                 |
| EVI      | Vegetation vigor in dense vegetation                |
| SAVI     | Vegetation response with soil-background adjustment |
| MSAVI    | Vegetation response with reduced soil influence     |
| NBR      | Vegetation disturbance / burn-related signal        |
| NDRE-B06 | Red-edge vegetation signal using B06                |
| NDRE-B07 | Red-edge vegetation signal using B07                |

The index implementations are located in:

```text
src/indices/
```

For example:

```bash
python -m src.indices.ndvi
```

Individual index modules can be run independently.

---

# Current Index Structure

```text
src/indices/
├── ndvi.py
├── ndre.py
├── gndvi.py
├── ndmi.py
├── evi.py
├── savi.py
├── msavi.py
├── nbr.py
├── ndre_b06.py
├── ndre_b07.py
└── palm_mask.py
```

The index calculations use the same underlying Sentinel-2 imagery so that different measurements can eventually be compared as model features.

---

# Palm Vegetation Mask

The project includes a simple palm vegetation mask in:

```text
src/indices/palm_mask.py
```

The current mask uses four vegetation indices:

* NDVI
* NDRE
* NDRE-B06
* NDRE-B07

A pixel is classified as a potential palm vegetation pixel only when it passes all four thresholds.

Current thresholds:

```python
NDVI_THRESHOLD = 0.65
NDRE_THRESHOLD = 0.45
NDRE_B06_THRESHOLD = 0.13
NDRE_B07_THRESHOLD = 0.039
```

The mask is intentionally kept simple at this stage.

It is a **spectral vegetation mask**, not a calibrated probability that a pixel contains an individual oil-palm tree.

This distinction is important because Sentinel-2 imagery has approximately 10 m spatial resolution, while individual palm crowns are generally smaller than a Sentinel-2 pixel. The current goal is therefore to identify areas with spectral characteristics consistent with healthy vegetation rather than to identify individual trees.

Run the palm mask with:

```bash
python -m src.indices.palm_mask
```

---

# Image Quality and SCL Filtering

Sentinel-2 Level-2A imagery includes a Scene Classification Layer (SCL).

The project uses SCL information to remove unusable pixels from processing.

The current valid-pixel workflow is designed to exclude conditions such as:

* Clouds
* Cloud shadows
* Water
* Other unusable pixels

This is particularly important when building monthly composites because individual satellite scenes can contain substantial cloud contamination.

---

# Monthly Composites

The project can combine multiple Sentinel-2 scenes into a monthly composite.

Run:

```bash
python -m src.composites.monthly_sentinel2_composite
```

The monthly workflow:

1. Searches for Sentinel-2 scenes.
2. Groups available imagery by month.
3. Processes individual scenes.
4. Applies SCL filtering.
5. Masks the imagery to the plantation.
6. Combines valid pixels from multiple dates.
7. Produces a monthly composite.
8. Tracks the number of valid observations for each pixel.

The purpose of the monthly composite is to reduce the influence of clouds and other unusable observations while creating a more representative monthly measurement.

---

# Scene Processing

Individual Sentinel-2 scenes are processed using:

```text
src/composites/process_sentinel2_scene.py
```

The scene-processing workflow handles:

* Sentinel-2 band retrieval
* SCL retrieval
* Reprojection
* Plantation geometry transformation
* Pixel masking
* Valid observation identification

This processing is reused by the monthly composite workflow.

---

# Visualization

The project includes tools for visually inspecting the satellite data.

Visualization code is located in:

```text
src/visuals/
```

Current visualization tools include:

```text
plantation_image.py
feature_comparison.py
```

## Plantation Image

Run:

```bash
python -m src.visuals.plantation_image
```

This creates visualizations of the selected plantation, including an RGB image and vegetation-index outputs.

Example outputs are saved in:

```text
outputs/
```

---

# Feature Comparison

The project also contains a feature comparison tool for examining multiple vegetation indices together.

This allows the same plantation area to be compared across different spectral measurements.

This is useful for determining whether different indices provide unique information or are measuring very similar vegetation characteristics.

---

# Feature Correlation

The project includes a correlation-analysis workflow in:

```text
src/analysis/feature_correlation.py
```

The correlation analysis is used to examine relationships between the satellite-derived features.

This is important for production modeling because several vegetation indices can contain very similar information.

For example, early testing showed strong correlations between:

* EVI and MSAVI
* EVI and SAVI
* SAVI and MSAVI
* NDMI and NBR
* NDVI and GNDVI

Other red-edge features showed lower correlations with several of the major vegetation indices.

This analysis will help determine which satellite-derived features should eventually be included in the production model.

---

# Configuration

Project settings are stored in:

```text
config/settings.py
```

The main settings include:

```python
GEOJSON_PATH = "data/boundaries/ANTANG.geojson"

IMAGE_DATE = "2026-07-20"

START_DATE = "2026-01-01"
END_DATE = "2026-12-31"
```

### `GEOJSON_PATH`

Controls which plantation boundary is being analyzed.

### `IMAGE_DATE`

Controls the date used for single-image processing.

### `START_DATE` and `END_DATE`

Control the date range used for searches and longer-term analysis.

---

# Project Structure

```text
Palm-Oil-Production-Model/
│
├── config/
│   └── settings.py
│
├── data/
│   └── boundaries/
│       ├── PT.geojson
│       ├── PT. AGRO GADING SEJAHTERA.geojson
│       └── ANTANG.geojson
│
├── logs/
│   └── monthly_sentinel2_composite.log
│
├── notebooks/
│
├── outputs/
│
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   └── sentinel2_data.py
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── best_sentinel2_date.py
│   │   └── monthly_sentinel2_search.py
│   │
│   ├── indices/
│   │   ├── __init__.py
│   │   ├── ndvi.py
│   │   ├── ndre.py
│   │   ├── gndvi.py
│   │   ├── ndmi.py
│   │   ├── evi.py
│   │   ├── savi.py
│   │   ├── msavi.py
│   │   ├── nbr.py
│   │   ├── ndre_b06.py
│   │   ├── ndre_b07.py
│   │   └── palm_mask.py
│   │
│   ├── visuals/
│   │   ├── __init__.py
│   │   ├── plantation_image.py
│   │   └── feature_comparison.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── feature_correlation.py
│   │
│   └── composites/
│       ├── __init__.py
│       ├── monthly_sentinel2_composite.py
│       └── process_sentinel2_scene.py
│
├── .env
├── .gitignore
├── README.md
└── requirements.py
```

---

# Running the Project

## Install dependencies

Install the required Python packages listed by the project.

The project uses packages including:

* Python
* NumPy
* Rasterio
* PySTAC Client
* Shapely
* Matplotlib
* Requests
* Other geospatial and data-processing libraries

---

## Run an individual index

For example:

```bash
python -m src.indices.ndvi
```

---

## Run the palm mask

```bash
python -m src.indices.palm_mask
```

---

## Search for the best Sentinel-2 image

```bash
python -m src.search.best_sentinel2_date
```

---

## Search Sentinel-2 imagery by month

```bash
python -m src.search.monthly_sentinel2_search
```

---

## Generate a monthly composite

```bash
python -m src.composites.monthly_sentinel2_composite
```

---

## Generate plantation visualizations

```bash
python -m src.visuals.plantation_image
```

---

# Current Status

## Working Prototype

The following components have been implemented and tested:

* Sentinel-2 Data Space connection
* Sentinel-2 scene search
* Cloud-cover filtering
* Sentinel-2 band retrieval
* GeoJSON plantation boundaries
* Plantation boundary masking
* Sentinel-2 SCL filtering
* Single-date satellite processing
* Multiple vegetation and moisture indices
* Palm vegetation masking
* Plantation visualization
* Feature correlation analysis
* Monthly Sentinel-2 scene searching
* Monthly satellite composites
* Valid observation tracking

The project is still a **working research prototype**. The satellite-derived features have not yet been calibrated against actual plantation production data.

---

# Production Modeling Plan

The eventual production model will connect satellite-derived features to actual oil-palm production measurements.

The planned structure is:

```text
Plantation / Block
        ↓
Satellite Imagery
        ↓
Satellite-Derived Features
        ↓
Monthly / Seasonal Feature Dataset
        ↓
Actual Production Data
        ↓
Statistical / Machine Learning Model
        ↓
Production Prediction
```

Potential satellite-derived features include:

* NDVI
* NDRE
* GNDVI
* NDMI
* EVI
* SAVI
* MSAVI
* NBR
* NDRE-B06
* NDRE-B07
* Monthly feature statistics
* Temporal changes
* Valid observation counts
* Other satellite-derived features added during development

Actual production measurements will eventually be used as the target variable.

---

# Important Modeling Considerations

## Satellite-derived vs. non-satellite variables

The current production-model feature set is intended to focus on variables that can be derived from satellite imagery.

Variables such as:

* Palm age
* Fertilizer application
* Pest observations
* Field management practices

may be important to production, but they should not be represented as satellite-derived variables unless a reliable method is developed to estimate them from imagery.

These variables can potentially be incorporated later as additional datasets if reliable data becomes available.

---

# Future Work

The next stages of development include:

### 1. Structured Feature Dataset

Convert monthly composite results into a structured dataset where each observation represents a plantation block and time period.

Example:

```text
Block | Month | NDVI | NDRE | NDMI | EVI | ... | Production
```

### 2. Additional Satellite Features

Investigate additional features that may provide useful information about plantation condition and production.

### 3. Temporal Analysis

Analyze how satellite-derived features change over time rather than relying only on individual images.

### 4. Production Data Integration

Add actual production measurements for the corresponding plantation blocks and time periods.

### 5. Feature Selection

Determine which satellite-derived variables provide useful information for predicting production while avoiding unnecessary redundancy between highly correlated indices.

### 6. Model Development

Test appropriate statistical and machine-learning approaches using the satellite-derived features and actual production data.

### 7. Model Evaluation

Evaluate the model using appropriate validation data and metrics to determine how accurately satellite-derived information can explain or predict oil-palm production.

---

# Limitations

Several limitations are currently important to keep in mind:

* Sentinel-2 imagery has approximately 10 m resolution for the primary bands used in this project.
* Individual oil-palm crowns cannot generally be treated as individual pixels.
* Clouds and atmospheric conditions can reduce usable observations.
* Different months may have substantially different numbers of usable satellite observations.
* Vegetation indices can be highly correlated with one another.
* The current palm mask is a spectral vegetation mask and has not yet been validated against field-level palm labels.
* The current satellite-derived features have not yet been calibrated against actual production data.
* Thresholds used by the palm mask are initial research thresholds and may require validation against field observations.

---

# Data and Security

Copernicus Data Space credentials are stored locally through environment variables.

Credentials should **not** be committed to GitHub.

---

# About

This project is being developed as a satellite-based agricultural analytics pipeline for oil-palm plantations.

The ultimate objective is to determine whether freely available Sentinel-2 satellite imagery can provide useful, measurable features for understanding and modeling oil-palm production at the plantation/block level.
