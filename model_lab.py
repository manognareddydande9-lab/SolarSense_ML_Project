import os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor,
    GradientBoostingClassifier, BaggingRegressor, BaggingClassifier)
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_CATALOG = {
    'Supervised': {
        'Regression': [
            'Linear Regression', 'Ridge Regression', 'Lasso Regression',
            'Decision Tree Regressor (CART)', 'Random Forest Regressor',
            'Bagging Regressor', 'Gradient Boosting Regressor', 'LightGBM Regressor'
        ],
        'Classification': [
            'Logistic Regression', 'Decision Tree Classifier (C4.5)',
            'Decision Tree Classifier (CART)', 'Random Forest Classifier',
            'Bagging Classifier', 'Gradient Boosting Classifier', 'LightGBM Classifier'
        ]
    },
    'Unsupervised': {
        'Clustering': ['K-Means', 'Mini-Batch K-Means', 'Hierarchical Clustering', 'Agglomerative Clustering', 'DBSCAN'],
        'Dimensionality Reduction': ['PCA']
    }
}

def catalog_flat():
    out = []
    for family, groups in MODEL_CATALOG.items():
        for group, names in groups.items():
            for name in names:
                out.append((family, group, name))
    return out

def make_estimator(name):
    if name == 'Linear Regression': return LinearRegression()
    if name == 'Ridge Regression': return Ridge(alpha=1.0)
    if name == 'Lasso Regression': return Lasso(alpha=0.001, max_iter=10000)
    if name == 'Decision Tree Regressor (CART)': return DecisionTreeRegressor(max_depth=12, random_state=42)
    if name == 'Random Forest Regressor': return RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
    if name == 'Bagging Regressor': return BaggingRegressor(n_estimators=80, random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting Regressor': return GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42)
    if name == 'LightGBM Regressor':
        if not LIGHTGBM_AVAILABLE: raise RuntimeError('LightGBM is not installed. Run: python -m pip install lightgbm')
        return LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=-1, random_state=42, verbosity=-1)
    if name == 'Logistic Regression': return LogisticRegression(max_iter=1000)
    if name in ('Decision Tree Classifier (C4.5)', 'Decision Tree Classifier (CART)'):
        # C4.5 is represented with an entropy/information-gain tree; CART uses Gini.
        criterion = 'entropy' if 'C4.5' in name else 'gini'
        return DecisionTreeClassifier(max_depth=12, criterion=criterion, random_state=42)
    if name == 'Random Forest Classifier': return RandomForestClassifier(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
    if name == 'Bagging Classifier': return BaggingClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    if name == 'Gradient Boosting Classifier': return GradientBoostingClassifier(n_estimators=120, random_state=42)
    if name == 'LightGBM Classifier':
        if not LIGHTGBM_AVAILABLE: raise RuntimeError('LightGBM is not installed. Run: python -m pip install lightgbm')
        return LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, verbosity=-1)
    raise ValueError(f'Unknown model: {name}')

def numeric_frame(df, target=None):
    work = df.drop(columns=[target], errors='ignore').copy()
    # Encode non-numeric columns consistently for the demo/model lab.
    for col in work.columns:
        if not pd.api.types.is_numeric_dtype(work[col]):
            work[col] = pd.factorize(work[col].astype(str))[0]
    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.select_dtypes(include=[np.number])
    return work

def train_supervised(df, target, model_name, task):
    if target not in df.columns:
        raise ValueError('Selected target column was not found in the dataset.')
    X = numeric_frame(df, target)
    if X.shape[1] == 0:
        raise ValueError('No numeric/codable feature columns are available.')
    y = df[target].copy()
    if task == 'Classification':
        if not pd.api.types.is_numeric_dtype(y):
            y = LabelEncoder().fit_transform(y.astype(str))
        if y.nunique() < 2:
            raise ValueError('Classification needs at least two target classes.')
    else:
        y = pd.to_numeric(y, errors='coerce')
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
    X = X.replace([np.inf, -np.inf], np.nan)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y if task=='Classification' and y.nunique()<20 else None)
    estimator = make_estimator(model_name)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', estimator)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    if task == 'Regression':
        metrics = {'R²': float(r2_score(yte, pred)), 'MAE': float(mean_absolute_error(yte, pred)), 'RMSE': float(np.sqrt(mean_squared_error(yte, pred)))}
    else:
        metrics = {'Accuracy': float(accuracy_score(yte, pred))}
    path = os.path.join(MODEL_DIR, 'selected_model.pkl')
    joblib.dump(pipe, path)
    joblib.dump({'model': model_name, 'task': task, 'target': target, 'features': X.columns.tolist()}, os.path.join(MODEL_DIR, 'selected_metadata.pkl'))
    return metrics, len(X), X.shape[1]

def run_unsupervised(df, model_name):
    X = numeric_frame(df)
    if X.shape[1] == 0: raise ValueError('No numeric/codable features are available.')
    X = SimpleImputer(strategy='median').fit_transform(X)
    X = StandardScaler().fit_transform(X)
    if model_name == 'K-Means': model = KMeans(n_clusters=3, random_state=42, n_init=10)
    elif model_name == 'Mini-Batch K-Means': model = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=10, batch_size=256)
    elif model_name in ('Hierarchical Clustering', 'Agglomerative Clustering'): model = AgglomerativeClustering(n_clusters=3)
    elif model_name == 'DBSCAN': model = DBSCAN(eps=0.8, min_samples=5)
    elif model_name == 'PCA':
        model = PCA(n_components=min(2, X.shape[1]))
        transformed = model.fit_transform(X)
        variance = [round(float(v), 4) for v in model.explained_variance_ratio_]
        return {'type': 'PCA', 'rows': len(X), 'components': transformed.shape[1], 'variance': variance}
    else: raise ValueError('Unknown unsupervised model.')
    labels = model.fit_predict(X)
    unique, counts = np.unique(labels, return_counts=True)
    return {'type': 'Clustering', 'rows': len(X), 'clusters': int(len(unique)), 'distribution': {str(int(k)): int(v) for k,v in zip(unique, counts)}}
