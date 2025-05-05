#!/usr/bin/env python3
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def main():
    # 1) 前処理済みデータの読み込み
    df = pd.read_csv('data/processed/processed_data.csv')

    # 2) 特徴量／ターゲットに分割
    X = df.drop('target', axis=1)
    y = df['target']

    # 3) データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 4) SMOTE オーバーサンプリング
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 5) モデル定義・学習
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res)

    # 6) 評価
    print("=== SMOTE + RandomForest 分類レポート ===")
    print(classification_report(y_test, model.predict(X_test), digits=4))

    # 7) モデル保存
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/keiba_rf_model_smote.pkl')
    print("SMOTE モデル saved to models/keiba_rf_model_smote.pkl")

if __name__ == "__main__":
    main()
