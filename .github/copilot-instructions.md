# AI Coding Agent Instructions: Algerian Forest Fire Prediction

## Project Overview
This is a machine learning classification project predicting forest fire occurrence in the Algerian region using environmental and weather data. The workflow consists of:
- **EDA.ipynb**: Multi-dataset exploration and preprocessing (3 regional datasets merged with region labels)
- **Model_Training.ipynb**: Feature engineering, train/test split, and model development (incomplete)

## Data Pipeline & Architecture

### Dataset Management
- **EDA.ipynb** loads three separate CSV sources:
  - `Algerian_forest_fires_dataset_UPDATE.csv` (primary training data)
  - `Bejaia Region ForestFire Dataset.csv` (region label: 0)
  - `Sidi-Bel Abbes Region ForestFire Dataset.csv` (region label: 1)
- **Model_Training.ipynb** uses cleaned output: `Algerian_forest_fires_dataset_CLEANED.csv`
- **Critical pattern**: Merge creates `dataset`, `dataset2`, `dataset3` variables; always reset_index after dropna()

### Feature Engineering Conventions
In Model_Training.ipynb:
- Drop temporal features: `day`, `month`, `year` (low predictive value)
- Target encoding pattern: `df["classes"] = np.where(df["Classes"].str.contains("not fire"), 0, 1)` (binary classification)
- Regression target: `FWI` (Fire Weather Index) for continuous prediction
- **X/Y split**: X excludes FWI, Y is FWI column

## Critical Incompleteness & Next Steps
1. **Model_Training.ipynb cell 13**: Import statement at line 44 executed with errors (likely import path issue)
2. **Empty cells 14-15**: Placeholder for train_test_split execution and model training
3. **Missing**: Actual model definitions, cross-validation, hyperparameter tuning

### Expected Workflow to Complete
```python
# After imports are fixed:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Then add models (typically sklearn regressors for FWI prediction)
# Common choices: RandomForest, XGBoost, LinearRegression
```

## Developer Workflow
- **Execute notebooks sequentially**: EDA first (establishes cleaned CSV), then Model_Training
- **Kernel state matters**: Variables persist across cells within same session
- **File paths**: All relative to notebook directory (assumes CSVs in root folder)
- **Matplotlib**: Uses `%matplotlib inline` for inline visualization

## Project-Specific Patterns
1. **Multi-dataset consolidation**: Prefix loading with region labels (0, 1, 2) for regional analysis
2. **String-based class detection**: Uses `.str.contains()` rather than exact matching (handles variations in "not fire")
3. **Numpy conditional encoding**: Prefer `np.where()` over if-else loops for vectorized operations
4. **Value counts validation**: Always verify class distribution after encoding changes

## Common Integration Points
- **Pandas-Seaborn workflow**: EDA uses seaborn for distribution plots, assumes matplotlib inline mode
- **Sklearn imports**: train_test_split, preprocessing, model selection modules needed
- **CSV I/O**: pd.read_csv() relative paths; ensure working directory is notebook directory

## Debugging Notes
- If imports fail in Model_Training.ipynb: Verify sklearn is installed in kernel (`pip install scikit-learn`)
- If CLEANED.csv not found: Run EDA.ipynb first to generate it
- Class imbalance visible via `value_counts()` — may need stratified splitting or class weighting
