#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def main(input_csv):
    # データ読み込み
    df = pd.read_csv(input_csv)

    if 'pred_prob' not in df.columns:
        raise ValueError("CSVに 'pred_prob' カラムが存在しません")

    # ヒストグラムを描画
    plt.figure(figsize=(10, 6))
    plt.hist(df['pred_prob'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # 保存先パス
    output_path = os.path.join('data/results', 'pred_prob_hist.png')
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"ヒストグラム画像を {output_path} に保存しました")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot prediction probability histogram')
    parser.add_argument(
        '--input', required=True,
        help='予測CSVの相対パス (例: results/predictions.csv)'
    )
    args = parser.parse_args()
    main(args.input)
