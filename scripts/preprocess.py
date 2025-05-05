import pandas as pd
import os

# --- 1. データ読み込み ---
input_path = os.path.join("data", "raw", "race-result-horse.csv")
df = pd.read_csv(input_path)

# --- 2. finishing_position を数値型に変換 ---
# 文字列のままの行は NaN になるため、後で除外します
df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')

# 数値に変換できなかった行（中止や取消など）は除外
df = df.dropna(subset=['finishing_position'])

# 整数型にキャスト
df['finishing_position'] = df['finishing_position'].astype(int)

# --- 3. ターゲット変数生成（複勝圏） ---
df['target'] = (df['finishing_position'] <= 3).astype(int)

# --- 4. 不要な列を削除 ---
df = df.drop(['race_id', 'horse_id', 'horse_name', 'finishing_position'], axis=1)

# --- 5. カテゴリ変数のエンコーディング ---
df = pd.get_dummies(df, columns=['jockey', 'trainer'], drop_first=True)

# --- 6. 欠損値処理（平均値補完） ---
df = df.fillna(df.mean(numeric_only=True))

# --- 7. 処理後データの保存 ---
output_path = os.path.join("data", "processed", "processed_data.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
