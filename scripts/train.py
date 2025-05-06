#!/usr/bin/env python3
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def main():
    # 1) データ読み込み
    df = pd.read_csv('data/processed/processed_data.csv')
    X  = df.drop('target', axis=1)
    y  = df['target']

    # 2) データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) パイプライン定義：SMOTE → RandomForest
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # 4) 探索するハイパーパラメータグリッド
    param_grid = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_leaf': [1, 5, 10],
        'rf__class_weight': ['balanced', {'1':2, '0':1}]
    }

    # 5) GridSearchCV（評価指標はRecallを重視）
    gs = GridSearchCV(
        pipeline,
        param_grid,
        scoring='f1',     # 複勝圏クラスの F1 を最大化
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    gs.fit(X_train, y_train)

    # 6) ベストモデルで再評価
    best = gs.best_estimator_
    print(">> Best parameters:", gs.best_params_)
    y_pred = best.predict(X_test)
    print("\n=== Tuned model classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    # 7) 保存
    os.makedirs('models', exist_ok=True)
    joblib.dump(best, 'models/keiba_rf_model_tuned.pkl')
    print("Tuned model saved to models/keiba_rf_model_tuned.pkl")

if __name__ == "__main__":
    main()
