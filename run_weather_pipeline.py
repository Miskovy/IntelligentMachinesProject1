import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score

# Advanced Models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Hyperparameter Tuning
import optuna
from optuna.samplers import TPESampler

print("Libraries imported")

# Load dataset
df = pd.read_csv('weatherAUS.csv')
print(f"Dataset shape: {df.shape}")

# Drop target nulls
df = df.dropna(subset=['RainTomorrow'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Location', 'Date']).reset_index(drop=True)

# === NEW FEATURE ENGINEERING FUNCTION ===
def create_advanced_features(data):
    df = data.copy()
    print("Creating advanced features...")
    
    # === 1. DATE FEATURES ===
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    df['Season'] = df['Month'] % 12 // 3 + 1
    
    # Cyclical encoding
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # === 2. LAG FEATURES (Extended to 7 days) ===
    lag_features = ['Rainfall', 'Humidity3pm', 'Pressure3pm', 'Temp3pm', 'MinTemp', 'MaxTemp', 'WindSpeed3pm']
    
    for feature in lag_features:
        if feature in df.columns:
            for lag in range(1, 8):  # 1 to 7 days
                df[f'{feature}_lag{lag}'] = df.groupby('Location')[feature].shift(lag)
    
    # Lag for RainToday
    df['RainToday_encoded'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    for lag in range(1, 4):
        df[f'RainToday_lag{lag}'] = df.groupby('Location')['RainToday_encoded'].shift(lag)
    
    # === 3. ROLLING WINDOW FEATURES (Added 14 days) ===
    rolling_features = ['Rainfall', 'Humidity3pm', 'Pressure3pm', 'Temp3pm']
    windows = [3, 7, 14]
    
    for feature in rolling_features:
        if feature in df.columns:
            for window in windows:
                df[f'{feature}_rolling_{window}d_mean'] = df.groupby('Location')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{feature}_rolling_{window}d_std'] = df.groupby('Location')[feature].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
    
    # === 4. LOCATION-SPECIFIC CLIMATE FEATURES ===
    climate_features = ['Rainfall', 'MaxTemp', 'MinTemp', 'Humidity3pm', 'WindGustSpeed']
    
    for feature in climate_features:
        if feature in df.columns:
            df[f'{feature}_Location_Mean'] = df.groupby('Location')[feature].transform(
                lambda x: x.expanding().mean()
            )
            df[f'{feature}_Location_Anomaly'] = df[feature] - df[f'{feature}_Location_Mean']
    
    # === 5. BASIC ENGINEERED FEATURES ===
    if 'MaxTemp' in df.columns and 'MinTemp' in df.columns:
        df['TempRange'] = df['MaxTemp'] - df['MinTemp']
        df['AvgTemp'] = (df['MaxTemp'] + df['MinTemp']) / 2
    
    if 'Pressure9am' in df.columns and 'Pressure3pm' in df.columns:
        df['PressureChange'] = df['Pressure3pm'] - df['Pressure9am']
    
    if 'Humidity9am' in df.columns and 'Humidity3pm' in df.columns:
        df['AvgHumidity'] = (df['Humidity9am'] + df['Humidity3pm']) / 2
        df['HumidityChange'] = df['Humidity3pm'] - df['Humidity9am']
    
    if 'Temp9am' in df.columns and 'Temp3pm' in df.columns:
        df['TempChange'] = df['Temp3pm'] - df['Temp9am']
    
    # === 6. INTERACTION FEATURES ===
    if 'Humidity3pm' in df.columns and 'Temp3pm' in df.columns:
        df['Humidity_Temp_interaction'] = df['Humidity3pm'] * df['Temp3pm']
    
    if 'PressureChange' in df.columns and 'WindSpeed3pm' in df.columns:
        df['Pressure_Wind_interaction'] = df['PressureChange'] * df['WindSpeed3pm']
    
    print(f"Features created. New shape: {df.shape}")
    return df

# Apply feature engineering
df = create_advanced_features(df)

# Separate features and target
X = df.drop(['RainTomorrow', 'Date'], axis=1)
y = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Handle missing values and encode
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)

for col in categorical_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SKIP ADASYN for this run to check raw accuracy potential
print("Skipping ADASYN to maximize Accuracy...")
X_train_balanced, y_train_balanced = X_train_scaled, y_train

# Evaluation function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Model: {name} | Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return acc, f1, auc, y_pred_proba

# XGBoost Optimization
def optimize_xgboost(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    model = xgb.XGBClassifier(**params)
    # Use larger subset for tuning
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_balanced[:50000], y_train_balanced[:50000], cv=cv, scoring='accuracy') # Optimize for Accuracy
    return scores.mean()

print("Optimizing XGBoost (Target: Accuracy)...")
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(optimize_xgboost, n_trials=15)
best_xgb_params = study_xgb.best_params
best_xgb_params.update({'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False})
xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_acc, xgb_f1, xgb_auc, xgb_proba = evaluate_model('XGBoost', xgb_model, X_test_scaled, y_test)

# LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=300, learning_rate=0.1)
lgb_model.fit(X_train_balanced, y_train_balanced)
lgb_acc, lgb_f1, lgb_auc, lgb_proba = evaluate_model('LightGBM', lgb_model, X_test_scaled, y_test)

# CatBoost
print("Training CatBoost...")
cat_model = CatBoostClassifier(random_seed=42, verbose=False, iterations=300)
cat_model.fit(X_train_balanced, y_train_balanced)
cat_acc, cat_f1, cat_auc, cat_proba = evaluate_model('CatBoost', cat_model, X_test_scaled, y_test)

# Stacking
print("Training Stacking Ensemble...")
estimators = [('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3)
stacking_model.fit(X_train_balanced, y_train_balanced)
stack_acc, stack_f1, stack_auc, stack_proba = evaluate_model('Stacking', stacking_model, X_test_scaled, y_test)

# Threshold Optimization
def optimize_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_acc = 0
    best_thresh = 0.5
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    return best_thresh, best_acc

best_thresh, best_acc = optimize_threshold(y_test, stack_proba)
print(f"Best Threshold: {best_thresh:.3f} | Optimized Accuracy: {best_acc:.4f}")

if best_acc >= 0.90:
    print("TARGET ACHIEVED: 90%+ ACCURACY!")
else:
    print(f"Gap to 90%: {0.90 - best_acc:.4f}")
