# Finding Donors for Charity Organization

This project aims to predict whether an individual earns more than $50,000 annually using data from the 1994 U.S. Census. The predictions can help non-profit organizations identify potential donors and optimize their outreach efforts. The project employs supervised learning techniques, ensemble methods, and SHAP (SHapley Additive exPlanations) for model interpretability.

## Features

- **Data Preprocessing**:
  - Logarithmic transformation of skewed features.
  - One-hot encoding for categorical variables.
  - Normalization of numerical features.
- **Model Training**:
  - Implementation of various supervised learning algorithms, including:
    - Decision Trees
    - Logistic Regression
    - Support Vector Machines (SVM)
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Linear Discriminant Analysis
    - Multinomial Naive Bayes
  - Comparison of models based on accuracy, F1-score, and training/testing time.
- **Model Optimization**:
  - Use of ensemble methods like Random Forest, AdaBoost, and XGBoost.
  - Hyperparameter tuning using GridSearchCV.
- **Model Interpretability**:
  - SHAP values to explain feature importance and interactions.
  - Visualization of feature contributions and dependence plots.

## Requirements

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `xgboost`
  - `shap`

## Installation

1. Clone this repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost shap



```markdown
# Finding Donors for Charity Organization

This project aims to predict whether an individual earns more than $50,000 annually using data from the 1994 U.S. Census. The predictions can help non-profit organizations identify potential donors and optimize their outreach efforts. The project employs supervised learning techniques, ensemble methods, and SHAP (SHapley Additive exPlanations) for model interpretability.

## Features

- **Data Preprocessing**:
  - Logarithmic transformation of skewed features.
  - One-hot encoding for categorical variables.
  - Normalization of numerical features.
- **Model Training**:
  - Implementation of various supervised learning algorithms, including:
    - Decision Trees
    - Logistic Regression
    - Support Vector Machines (SVM)
    - K-Nearest Neighbors (KNN)
    - Gaussian Naive Bayes
    - Linear Discriminant Analysis
    - Multinomial Naive Bayes
  - Comparison of models based on accuracy, F1-score, and training/testing time.
- **Model Optimization**:
  - Use of ensemble methods like Random Forest, AdaBoost, and XGBoost.
  - Hyperparameter tuning using GridSearchCV.
- **Model Interpretability**:
  - SHAP values to explain feature importance and interactions.
  - Visualization of feature contributions and dependence plots.

## Requirements

- Python 3.x
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `xgboost`
  - `shap`

## Installation

1. Clone this repository or download the project files.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost shap
   ```

## Dataset

The dataset used in this project is the [Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income) from the UCI Machine Learning Repository. It contains demographic and employment-related attributes for individuals, along with their income labels (`<=50K` or `>50K`).

### Features

- **Numerical**: `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Target**: `income` (`<=50K` or `>50K`)

## Usage

1. Open the Jupyter Notebook file `Copy_of_Finding_Donors_Charity + Shap.ipynb`.
2. Run the cells sequentially to:
   - Preprocess the data.
   - Train and evaluate multiple supervised learning models.
   - Optimize the best-performing model.
   - Interpret the model using SHAP.

## Examples



### Data Preprocessing
Logarithmic transformation of skewed features:
```python
data['capital-gain_log'] = np.log((data['capital-gain']) + 1)
data['capital-loss_log'] = np.log((data['capital-loss']) + 1)
```

One-hot encoding of categorical variables:
```python
data_cat_transformed = pd.get_dummies(data_log_transformed, drop_first=True)
```

### Model Training and Evaluation
Training and evaluating a Decision Tree classifier:
```python
clf_DT = DecisionTreeClassifier(random_state=0)
results, report = train_predict(clf_DT)
print(results)
print(report)
```

### SHAP Interpretability
Generating SHAP values and visualizing feature importance:
```python
shap_values = xgb_explainer.shap_values(X_train, y_train)
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="bar")
```

## Results

- **Best Model**: XGBoost achieved the highest performance with optimized hyperparameters.
- **Key Features**: SHAP analysis revealed that features like `marital-status`, `age`, and `education-num` had the most significant impact on predictions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by the UCI Machine Learning Repository.
- SHAP library for model interpretability.
- Scikit-learn and XGBoost for machine learning algorithms.
```
