# RapidAPI Success Analyzer

A Python script to analyze API marketplace data and identify key factors for API success.

## Setup & Usage

Sync environment | Install dependencies | Run script:

```bash
uv sync
uv run main.py
```

## Analysis Steps

The script performs the following analysis:

1. **Exploratory Analysis**: Shows top APIs and categories by a calculated "Success Score".
2. **Model Training**: Trains ML models to predict API success.
3. **Feature Importance**: Identifies the most influential features for success.
4. **Cluster Analysis**: Automatically finds the optimal number of API clusters and groups them by their characteristics.

## Output Files

- `feature_importance.png`: Chart of the most important features.
- `feature_importance_scores.csv`: The raw data for the feature importance scores.
- `elbow_curve.png`: Plot used to determine the optimal number of clusters.
- `cluster_visualizations.png`: PCA and UMAP plots showing the identified API clusters.
