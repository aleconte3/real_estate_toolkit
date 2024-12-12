# Real Estate Toolkit

This repository contains a Python-based toolkit for analyzing and predicting real estate prices. The project utilizes machine learning models to predict house prices based on various features of the real estate data. It also includes data cleaning and feature preparation functionality.

## Project Overview

The **Real Estate Toolkit** aims to assist in analyzing real estate data, cleaning it, preparing the features, training machine learning models, and making predictions about house prices.

### Key Features:
- **Data Cleaning**: Handles missing values, renames columns to follow `snake_case` convention, and prepares data for analysis.
- **Feature Preparation**: Selects relevant features, normalizes them, and prepares them for use in machine learning models.
- **Machine Learning Models**: Includes basic models like **Linear Regression**, **Random Forest**, and **Gradient Boosting**.
- **Predictions**: Forecasts house prices using the trained models and saves predictions to a CSV file.

## Installation

### Requirements:
- Python 3.8+
- Poetry (for dependency management)
  
You can install the dependencies using Poetry by running:

```bash
poetry install