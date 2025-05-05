import pandas as pd
import numpy as np
import os

# --- 1. データ読み込み ---
input_path = os.path.join("data", "raw", "race-result-horse.csv")
df = pd.read_csv(input_path)

# --- 2. finishing_position を数値型に変換 ---
df['finishing_position'] = pd.to_numeric(df['finishing_position'], errors='coerce')
df = df.dropna(subset=['finishing_position'])
df['finishing_position'] = df['finishing_position'].astype(int)

# --- 3. ターゲット変数生成（複勝圏） ---
df['target'] = (df['finishing_position'] <= 3).astype(int)

# --- 4. finish_time を秒に変換 ---
def to_sec(t):
    try:
        m, s = t.split(':')
        return float(m) * 60 + float(s)
    except:
        return np.nan

df['finish_time'] = df['finish_time'].apply(to_sec)

# --- 5. length_behind_winner を float に変換 ---
def to_length(x):
    if pd.isna(x): 
        return np.nan
    if 'クビ' in x: 
        return 0.25
    if 'アタマ' in x: 
        return 0.2
    # 例: "3-3/4" → ["3","3","4"]
    parts = x.replace('-', ' ').replace('/', ' ').split()
    try:
        if len(parts) == 3:
            return float(parts[0]) + float(parts[1]) / float(parts[2])
        return float(parts[0])
    except:
        return np.nan

df['length_behind_winner'] = df['length_behind_winner'].apply(to_length)

# --- 6. running_position_* 列を一括削除 ---
drop_cols = [c for c in df.columns if c.startswith('running_position_')]
df = df.drop(columns=drop_cols)

# --- 7. 不要な列削除 ---
df = df.drop(['race_id', 'horse_id', 'horse_name', 'finishing_position'], axis=1)

# --- 8. カテゴリ変数のエンコーディング ---
df = pd.get_dummies(df, columns=['jockey', 'trainer'], drop_first=True)

# --- 9. 欠損値処理（平均値補完） ---
df = df.fillna(df.mean(numeric_only=True))

# --- 10. 保存 ---
output_path = os.path.join("data", "processed", "processed_data.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
