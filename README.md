# SolarSense — Solar Power Generation Prediction System

Professional Flask + Machine Learning project modeled on the overall presentation of the supplied Placement Prediction System reference.

## Kaggle source
Solar Power Generation Data by Ani Kannal:
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

The Kaggle dataset contains generation and weather-sensor data for two Indian solar plants at 15-minute intervals over 34 days.

**Important:** this ZIP contains a small DEMO dataset with the same column structure so it runs immediately. For your final college review, replace the two CSVs in `data/` with the original Kaggle files and run `py train.py`.

## Target
`AC_POWER` — solar plant AC power output.

## Models
Linear Regression, Ridge Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, and a Random Forest + Gradient Boosting Voting Ensemble.

## Metrics
R², MAE, RMSE.

## Run on Windows
```powershell
py -m pip install -r requirements.txt
py train.py
py app.py
```
Then open `http://127.0.0.1:5000`.

## Project flow
Dashboard → Analytics/EDA → Preprocessing → Model Training → Prediction → Upload Dataset → About/Help.

## Model Selection Lab (New)
The updated project includes a dedicated **Model Selection** page. From the sidebar, open **Model Selection** to choose:

- **Supervised → Regression:** Linear Regression, Ridge, Lasso, Decision Tree Regressor (CART), Random Forest, Bagging, Gradient Boosting, LightGBM.
- **Supervised → Classification:** Logistic Regression, Decision Tree Classifier (C4.5), Decision Tree Classifier (CART), Random Forest, Bagging, Gradient Boosting, LightGBM.
- **Unsupervised → Clustering:** K-Means, Mini-Batch K-Means, Hierarchical/Agglomerative, DBSCAN.
- **Unsupervised → Dimensionality Reduction:** PCA.

For supervised learning, you can choose the **Target Column** yourself. For unsupervised learning, target selection is hidden because these algorithms do not require a target column.

### Run
```powershell
cd "C:\Machine Learning\esd\SolarSense_ML_Project"
python -m pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

If LightGBM installation is not available, the LightGBM options will show an installation message. Install it with:
```powershell
python -m pip install lightgbm
```
