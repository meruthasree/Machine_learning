# Smart Wardrobe Assistant

## Project Overview

This project is a machine learning application for fashion and wardrobe assistance. It explores fashion dataset contents, image metadata, and style information to build models or analysis that can help recommend outfits, identify styles, and support a smart wardrobe assistant.

## What is included

- `Smart Wardrobe Assistant.IPYNB` - Main notebook containing the analysis, data processing, model exploration, or prototype workflow for the smart wardrobe assistant.
- `fashion-dataset/` - A dataset folder containing fashion-related CSV files and JSON metadata for images, styles, and wardrobe items.
- `styles.csv` - A CSV file likely containing style labels or fashion categorization information used in the project.


## Dataset Structure

The dataset is organized with image and style metadata, including:

- `fashion-dataset/images.csv` and `fashion-dataset/styles.csv`
- `fashion-dataset/images/` - likely contains image files or metadata references
- `fashion-dataset/styles/` - contains style JSON files for individual fashion items

This structure supports tasks such as:

- fashion item classification
- style recommendation
- outfit matching and combination analysis
- metadata-driven wardrobe suggestions

## Goals

The main goals of this project are to:

- analyze fashion dataset contents and structure
- build a foundation for wardrobe recommendation or fashion classification models
- create a smart assistant workflow that uses styles and image data to suggest clothing

## Usage

Open `Smart Wardrobe Assistant.IPYNB` in Jupyter Notebook or JupyterLab to view the analysis and run the project steps. The notebook should contain data loading, exploration, and any modeling or recommendation logic.

## Workflow

The notebook follows a simple but complete pipeline:

1. Start with the raw fashion data in `styles.csv`.
2. Inspect the major patterns in master category, usage, and season.
3. Clean the records, keep apparel items, fill missing values, and encode the categorical fields.
4. Train occasion predictors with Logistic Regression and Random Forest, using SMOTE to reduce class imbalance.
5. Learn outfit pairing rules with a Decision Tree that predicts the most suitable bottomwear for each topwear item.
6. Score outfit combinations with Linear Regression, Ridge Regression, and Lasso Regression on engineered pairwise features.
7. Bring the models together with the custom `color_harmony` rule to generate the final wardrobe recommendations.

## Models and Algorithms Used

The notebook combines a few different ideas instead of relying on a single model:

- `LabelEncoder` turns categories like gender, usage, season, colour, master category, subcategory, and article type into numeric form.
- `StandardScaler` keeps the feature scale consistent before classification and regression.
- `SMOTE` balances the occasion labels so rare classes do not get ignored.
- `Logistic Regression` predicts the usage or occasion of an item.
- `Random Forest Classifier` provides a second occasion-classification model for comparison.
- `Decision Tree Classifier` handles outfit matching by predicting the best bottomwear for a topwear item.
- `PolynomialFeatures` expands the engineered pair features before scoring.
- `Linear Regression`, `Ridge Regression`, and `Lasso Regression` estimate outfit quality and let you compare model behavior.
- The custom `color_harmony` rule adds the styling logic by grouping colours into neutral, warm, and cool families.

In short, the project is metadata-driven: it filters apparel items, checks category, season, usage, and gender compatibility, then blends machine learning predictions with a styling heuristic to recommend outfits.

## Evaluation Metrics

The output is checked based on matching relevance, category accuracy, and recommendation quality. The main focus is whether the suggested clothing combinations are practical and logically correct.

## How to Run
Open `Smart Wardrobe Assistant.IPYNB`
Keep fashion-dataset/ in the correct folder
Install required libraries (pandas, numpy, matplotlib, scikit-learn)
Run the notebook cells step by step
View the generated outfit suggestions and grouped results
