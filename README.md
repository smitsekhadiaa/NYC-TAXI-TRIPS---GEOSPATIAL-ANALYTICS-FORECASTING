# Geospatial NYC Taxi Trips

A geospatial analytics and machine learning project built on NYC taxi trip data to study demand patterns, trip behavior, spatial clustering, anomaly detection, and fare/ETA forecasting.

## Project Overview

This project processes NYC taxi trip data and transforms it into a set of analytics products exposed through:

- A Flask backend for APIs and training workflows
- A Streamlit frontend for interactive visualization
- Data processing and analysis pipelines for feature generation
- Machine learning modules for clustering, anomaly detection, forecasting, and trip pattern mining

The system is designed to support both exploratory analysis and application-style consumption of generated insights.

## Objectives

The main goals of the project are:

- Analyze spatial and temporal patterns in NYC taxi trips
- Understand borough and zone-level trip activity
- Detect unusual or anomalous trips
- Cluster trip demand behavior by time segment
- Forecast trip fare and ETA for selected routes
- Surface results through an interactive frontend dashboard

## Key Features

### 1. Data Processing
The project includes preprocessing pipelines that convert raw source data into structured, analysis-ready datasets. These processed datasets are exported into CSV files for downstream analytics, backend services, and model training.

### 2. Exploratory Data Analysis
The repository contains utilities to generate descriptive statistics, correlation analysis, and summary artifacts that help understand trip volume, distance, fare, and temporal patterns.

### 3. Clustering
Clustering models are trained on trip behavior across multiple time segments such as morning, afternoon, evening, night, and all-day traffic. The backend serves clustering payloads used by the frontend to render map-based spatial insights.

### 4. Anomaly Detection
The anomaly detection pipeline identifies unusual trips such as extreme-speed rides and fare outliers. Generated summaries and outputs are exposed through backend APIs and visualized in the frontend.

### 5. Fare and ETA Forecasting
The forecasting pipeline trains machine learning models to estimate trip fare and trip duration using route, date, time, and location-based features. These models are served through prediction APIs.

### 6. Trip Pattern Mining
The project includes association-rule mining over trip data to identify recurring trip patterns and frequently observed route combinations.

### 7. Interactive Frontend
A Streamlit application provides a user-facing dashboard for:
- high-level analytics
- clustering exploration
- fare and ETA prediction
- anomaly and trip pattern insights

## Project Architecture

The project is organized into three main layers:

- `data/`  
  Contains raw sources, processed datasets, exported CSVs, analysis outputs, and ML artifacts

- `backend/`  
  Flask application with API routes, CLI commands, and service modules for loading data, training models, and serving predictions/insights

- `frontend/`  
  Streamlit application with dashboard views, repositories, and UI components for consuming backend APIs and local artifacts

## Tech Stack

- Python
- Flask
- Streamlit
- Pandas
- NumPy
- DuckDB
- PySpark
- GeoPandas
- SQLAlchemy
- PyMySQL
- Scikit-learn
- Plotly
- PyDeck
- Matplotlib

## Repository Structure

```text
backend/                 Flask backend and ML/data services
frontend/                Streamlit frontend
data/
  raw_src_data/          Raw source files and shapefiles
  processed_data/        Processing logic
  processed_csv/         Exported CSV datasets
  data_analysis/         Statistics and EDA artifacts
  clustering_artifacts/  Clustering outputs and plots
  anomaly_artifacts/     Anomaly detection outputs
  forecasting_artifacts/ Forecasting models and metrics
  trip_pattern_artifacts/ Association rule mining outputs
docs/                    Architecture diagrams and supporting documentation
tools/                   Utility scripts
