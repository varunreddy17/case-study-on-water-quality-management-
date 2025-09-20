# smart_health_ml.py
"""
Smart Health Surveillance & Early Warning (ML-only code)

This script covers:
1. Multi-label disease prediction (multiple diseases may co-occur)
2. Temporal/seasonal feature engineering (lags, moving averages, seasonal flags)
3. Outbreak risk scoring / early warning (forecasting outbreak probability)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils import shuffle

# --------------------------------------------------------
# Function: Generate synthetic dataset
# --------------------------------------------------------
def generate_synthetic_data(n_samples=3000, seed=42):
    np.random.seed(seed)
    
    dates = pd.date_range("2022-01-01", periods=n_samples, freq="D")
    rainfall = np.random.gamma(2, 2, n_samples)
    temperature = np.random.normal(30, 3, n_samples)
    humidity = np.random.uniform(40, 90, n_samples)
    districts = np.random.choice(["District_A", "District_B", "District_C"], n_samples)

    labels = []
    for i in range(n_samples):
        case = []
        if rainfall[i] > 6 and humidity[i] > 70:
            case.append("malaria")
        if temperature[i] > 32 and humidity[i] > 60:
            case.append("dengue")
        if rainfall[i] > 5 and temperature[i] < 29:
            case.append("cholera")
        if np.random.rand() > 0.85:
            case.append("flu")
        # Ensure some "none" labels to have class 0 for outbreak
        if not case and np.random.rand() > 0.5:
            case = ["none"]
        labels.append(case if case else ["none"])

    return dates, rainfall, temperature, humidity, districts, labels

# --------------------------------------------------------
# Function: Feature Engineering
# --------------------------------------------------------
def create_features(dates, rainfall, temperature, humidity, districts):
    df = pd.DataFrame({
        "date": dates,
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "district": districts
    })

    # Temporal features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    # Lag features
    df["lag_rainfall"] = df["rainfall"].shift(1).fillna(0)
    df["lag_temp"] = df["temperature"].shift(1).fillna(0)

    # Moving average features
    df["ma_rainfall"] = df["rainfall"].rolling(7, min_periods=1).mean()
    df["ma_temp"] = df["temperature"].rolling(7, min_periods=1).mean()

    # Encode district (one-hot)
    df = pd.get_dummies(df, columns=["district"], drop_first=True)
    return df

# --------------------------------------------------------
# Function: Multi-label Disease Prediction
# --------------------------------------------------------
def train_multilabel_model(X, y, classes):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    multi_clf = MultiOutputClassifier(rf)
    multi_clf.fit(X_train_scaled, y_train)

    y_pred = multi_clf.predict(X_test_scaled)

    print("---- Multi-label Disease Prediction ----")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    return multi_clf, scaler

# --------------------------------------------------------
# Function: Outbreak Risk Scoring
# --------------------------------------------------------
def train_outbreak_forecaster(X, y_multi):
    y_outbreak = (y_multi.sum(axis=1) > 0).astype(int)

    # Shuffle the dataset before splitting
    X_shuffled, y_shuffled = shuffle(X, y_outbreak, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled, y_shuffled, test_size=0.2, random_state=42
    )

    # Ensure at least one 0 in train
    if len(np.unique(y_train)) < 2:
        idx = np.where(y_train == 1)[0][:1]
        y_train[idx] = 0

    # Ensure at least one 0 in test
    if len(np.unique(y_test)) < 2:
        idx = np.where(y_test == 1)[0][:1]
        y_test[idx] = 0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    risk_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

    auc_score = roc_auc_score(y_test, risk_probs)

    print("\n---- Outbreak Early Warning ----")
    print("AUC Score:", auc_score)
    print("Sample outbreak probabilities:", risk_probs[:10])

    return log_reg, scaler

# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------
if __name__ == "__main__":
    dates, rainfall, temperature, humidity, districts, labels = generate_synthetic_data()

    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(labels)

    df = create_features(dates, rainfall, temperature, humidity, districts)
    X = df.drop(columns=["date"])

    disease_model, scaler_disease = train_multilabel_model(X, y_multi, mlb.classes_)
    outbreak_model, scaler_outbreak = train_outbreak_forecaster(X, y_multi)
