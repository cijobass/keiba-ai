import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # ファイルパスの設定（相対パス指定）
    file_path = os.path.join("data", "results", "predictions.csv")

    # CSVの読み込み
    df = pd.read_csv(file_path)

    # pred_prob列の存在確認
    if 'pred_prob' not in df.columns:
        raise ValueError("'pred_prob' 列が predictions.csv に存在しません")

    # ヒストグラムの描画
    plt.figure(figsize=(10, 6))
    plt.hist(df["pred_prob"], bins=30, edgecolor="black", alpha=0.7, color="skyblue")
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")

    # 出力パスの設定と保存
    output_path = os.path.join("data", "results", "pred_prob_hist.png")
    plt.savefig(output_path)
    plt.show()
    print(f"✔ ヒストグラムを保存しました: {output_path}")

if __name__ == "__main__":
    main()
