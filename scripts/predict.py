#!/usr/bin/env python3
import argparse
import joblib
import json
import pandas as pd
import os

def main(input_csv, output_csv):
    # このスクリプトのディレクトリ
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # １つ上がプロジェクトルート
    root_dir   = os.path.abspath(os.path.join(script_dir, os.pardir))

    # 1) 学習済みモデルと閾値のロード
    model_path  = os.path.join(root_dir, 'models', 'keiba_rf_model.pkl')
    thresh_path = os.path.join(root_dir, 'models', 'threshold.json')
    model       = joblib.load(model_path)
    info        = json.load(open(thresh_path))
    threshold   = info['best_threshold']

    # 2) 新データ読み込み
    input_path  = os.path.join(root_dir, input_csv)
    df          = pd.read_csv(input_path)

    # 3) 予測処理
    X_new       = df.drop('target', axis=1, errors='ignore')
    proba       = model.predict_proba(X_new)[:, 1]
    pred        = (proba >= threshold).astype(int)

    # 4) 出力ディレクトリ作成＆結果保存
    output_path = os.path.join(root_dir, output_csv)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df['pred_prob']   = proba
    df['pred_target'] = pred
    df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keiba AI Predict')
    parser.add_argument(
        '--input',  required=True,
        help='入力CSVのプロジェクトルートからの相対パス (例: data/processed/processed_data.csv)'
    )
    parser.add_argument(
        '--output', required=True,
        help='出力CSVのプロジェクトルートからの相対パス (例: results/predictions.csv)'
    )
    args = parser.parse_args()
    main(args.input, args.output)
