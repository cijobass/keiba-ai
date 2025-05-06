#!/usr/bin/env python3
import argparse
import joblib
import json
import pandas as pd
import os

def main(input_csv, output_csv, model_path, thresh_path):
    # このスクリプトのディレクトリ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # １つ上がプロジェクトルート
    root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 学習済みモデルと閾値のロード
    model_full_path = os.path.join(root_dir, model_path)
    threshold_full_path = os.path.join(root_dir, thresh_path)
    model = joblib.load(model_full_path)

    with open(threshold_full_path) as f:
        info = json.load(f)

    # 古い形式 or 新形式 の両方に対応
    threshold = info.get('best_threshold', info.get('threshold'))
    if threshold is None:
        raise KeyError("threshold.json に 'threshold' もしくは 'best_threshold' キーが見つかりません")

    # 新データ読み込み
    input_path = os.path.join(root_dir, input_csv)
    df = pd.read_csv(input_path)

    # 特徴量を用意（target 列があれば除去）
    X_new = df.drop('target', axis=1, errors='ignore')

    # モデルを使った予測（確率とクラス）
    proba = model.predict_proba(X_new)[:, 1]
    pred = (proba >= threshold).astype(int)

    # 結果を元データに追加
    df['pred_prob'] = proba
    df['pred_target'] = pred

    # 出力ディレクトリ作成＆結果保存
    output_path = os.path.join(root_dir, output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keiba AI Predict')
    parser.add_argument('--input', required=True, help='入力CSVのパス（例: data/processed/processed_data.csv）')
    parser.add_argument('--output', required=True, help='出力CSVのパス（例: results/predictions.csv）')
    parser.add_argument('--model', default='models/keiba_rf_model_tuned.pkl', help='モデルファイルのパス（例: models/keiba_rf_model_tuned.pkl）')
    parser.add_argument('--threshold', default='models/threshold.json', help='閾値ファイルのパス（例: models/threshold.json）')

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.threshold)
