#!/usr/bin/env python3
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

def main(model_path: str, data_path: str, output_path: str,
         test_size: float, random_state: int):
    # 1) モデルとデータのロード
    model = joblib.load(model_path)
    df    = pd.read_csv(data_path)

    # 2) 特徴量／ターゲット分割
    X = df.drop('target', axis=1)
    y = df['target']

    # 3) 訓練／テスト分割（学習時と同じ設定で）
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 4) 確率予測
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5) Precision–Recall curve から最適閾値を探索（F1最大化）
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    # thresholds の長さは precision,recall より 1 小さいので合わせる
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx       = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    # 6) JSON で保存
    out = {
        "model":       model_path,
        "data":        data_path,
        "threshold":   best_threshold,
        "f1_score":    float(f1_scores[best_idx])
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved threshold = {best_threshold:.4f} (F1 = {f1_scores[best_idx]:.4f}) → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune probability threshold to maximize F1 and save to JSON"
    )
    parser.add_argument(
        "--model", "-m",
        default="models/keiba_rf_model_tuned.pkl",
        help="学習済みモデルのパス (pickle)"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/processed/processed_data.csv",
        help="前処理済みデータのCSVパス"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/threshold.json",
        help="出力する閾値JSONのパス"
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.2,
        help="テストデータに回す割合"
    )
    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=42,
        help="乱数シード (train_test_split 用)"
    )
    args = parser.parse_args()
    main(
        model_path=args.model,
        data_path=args.data,
        output_path=args.output,
        test_size=args.test_size,
        random_state=args.random_state
    )
