#!/usr/bin/env python3
import sys, os
import argparse

# ——— プロジェクトルートをパスに追加 ———
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
# —————————————————————————————

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from scraper.race_result_selenium import get_race_result
from scraper.race_entry_scraper   import get_race_entry

# --- 前処理ヘルパー --- 
def to_sec(t):
    try:
        m, s = t.split(':')
        return float(m) * 60 + float(s)
    except:
        return np.nan

def to_length(x):
    if pd.isna(x): 
        return np.nan
    if 'クビ' in x: 
        return 0.25
    if 'アタマ' in x: 
        return 0.2
    parts = x.replace('-', ' ').replace('/', ' ').split()
    try:
        if len(parts) == 3:
            return float(parts[0]) + float(parts[1]) / float(parts[2])
        return float(parts[0])
    except:
        return np.nan

def preprocess_entry(df: pd.DataFrame, model_features):
    """
    df: get_race_entry() が返す DataFrame（index=horse_number）
    model_features: model.feature_names_in_
    """
    # 1) 英語カラム名に揃える
    df = df.rename(columns={
        'frame':           'draw',
        'weight':          'actual_weight',
        'declared_weight': 'declared_horse_weight',
        # jockey, trainer, win_odds は既に 'jockey','trainer','win_odds'
    })

    # 2) 必要な数値列を追加
    df['length_behind_winner'] = np.nan
    df['finish_time']         = np.nan

    # 3) win_odds はすでに英語カラムなのでそのまま数値変換
    df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')

    # 4) jockey/trainer の one-hot
    df = pd.get_dummies(df, columns=['jockey','trainer'], drop_first=True)

    # 5) モデルが期待する列順に合わせ、足りない列は 0 埋め
    df = df.reindex(columns=model_features, fill_value=0)
    return df

def main(race_id, model_path, thresh_path):
    # --- 1) モデル・閾値ロード ---
    model     = joblib.load(model_path)
    info      = json.load(open(thresh_path))
    threshold = info.get('best_threshold', info.get('threshold'))

    # --- 2) 出馬表取得 & 前処理 ---
    entry_df = get_race_entry(race_id)  # index=horse_number の DataFrame を返す想定
    X_live   = preprocess_entry(entry_df, model.feature_names_in_)

    # ─── 数値変換 ───
    for col in ['actual_weight', 'declared_horse_weight']:
        X_live[col] = pd.to_numeric(X_live[col], errors='coerce')
    X_live = X_live.fillna(0)
    # ─────────────────

    # --- 3) 予測 ---
    proba = model.predict_proba(X_live)[:, 1]
    pred  = (proba >= threshold).astype(int)
    entry_df['pred_prob']   = proba
    entry_df['pred_target'] = pred

    # --- 4) レース結果取得 & ラベル付与 ---
    result_df = get_race_result(race_id).set_index('horse_number')
    result_df['target'] = (result_df['rank'].astype(int) <= 3).astype(int)

    # --- 5) 予測結果と実結果をマージ & 評価 ---
    merged = entry_df.join(
        result_df[['target','rank']],
        how='inner'
    )
    precision = precision_score(merged['target'], merged['pred_target'])
    recall    = recall_score(merged['target'], merged['pred_target'])
    f1        = f1_score(merged['target'], merged['pred_target'])

    # --- 結果表示 ---
    print(f"=== Race {race_id} Evaluation ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(merged[['pred_prob','pred_target','target','rank']])

    # 必要なら CSV 出力
    os.makedirs('data/results', exist_ok=True)
    merged.to_csv(f"data/results/live_{race_id}_eval.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Race Prediction & Evaluation")
    parser.add_argument('--race-id',   required=True, help="netkeiba race_id (例: 202405030712)")
    parser.add_argument('--model',     default='models/keiba_rf_model_tuned.pkl', help="モデルパス")
    parser.add_argument('--threshold', default='models/threshold.json',           help="閾値JSONパス")
    args = parser.parse_args()

    root        = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    model_path  = os.path.join(root, args.model)
    thresh_path = os.path.join(root, args.threshold)

    main(args.race_id, model_path, thresh_path)
