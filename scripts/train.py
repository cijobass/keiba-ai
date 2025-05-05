#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib, os

def main():
    # 1) 前処理済みデータ読み込み
    df = pd.read_csv('data/processed/processed_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']

    # 2) データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) モデル学習
    model = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4) 評価
    print(classification_report(y_test, model.predict(X_test), digits=4))

    # 5) 保存
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/keiba_rf_model_balanced.pkl')
    print("Balanced model saved to models/keiba_rf_model_balanced.pkl")

if __name__ == '__main__':
    main()
