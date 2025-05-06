#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def main():
    # パス設定
    input_path = os.path.join('data', 'results', 'predictions.csv')

    # データ読み込み
    df = pd.read_csv(input_path)

    # 正解ラベルと予測値
    y_true = df['target']
    y_pred = df['pred_target']
    y_prob = df['pred_prob']

    # 評価指標計算
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    roc_auc   = roc_auc_score(y_true, y_prob)
    cm        = confusion_matrix(y_true, y_pred)

    # 結果表示
    print("=== 📊 モデル評価結果 ===")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"ROC AUC Score  : {roc_auc:.4f}\n")

    print("=== 🔢 混同行列 ===")
    print(cm)

if __name__ == '__main__':
    main()
