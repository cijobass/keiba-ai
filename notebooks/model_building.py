# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: keiba-ai-env
#     language: python
#     name: python3
# ---

# %% [markdown]
# 1 ライブラリのインポート

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# %% [markdown]
# 2 データ読み込みと確認

# %%
# 前処理済みデータの読み込み
df = pd.read_csv('../data/processed/processed_data.csv')

# 基本情報を確認
display(df.head())
print(df.info())
print(df['target'].value_counts(normalize=True))

# %% [markdown]
# 3 特徴量／ターゲット分離とデータ分割

# %%
# 特徴量とターゲット
X = df.drop('target', axis=1)
y = df['target']

# 80:20 で訓練／テストに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape, y_train.value_counts())
print("Test: ", X_test.shape, y_test.value_counts())

# %% [markdown]
# 4 モデル構築・学習

# %%
# ランダムフォレストを定義・学習
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# %% [markdown]
# 5 予測と評価

# %%
# テストデータで予測
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 分類レポート
print(classification_report(y_test, y_pred, digits=4))

# 混同行列を描画
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Out','In']); ax.set_yticklabels(['Out','In'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j]>cm.max()/2 else 'black')
fig.tight_layout()
plt.show()


# %% [markdown]
# # ────────────────
# # 閾値調整による Recall 向上
# # ────────────────

# %%
from sklearn.metrics import precision_recall_curve
import numpy as np

# %% [markdown]
# # 1. 複勝確率を再利用
# # y_proba は上のセルですでに定義済み

# %% [markdown]
# # 2. Precision–Recall curve の計算

# %%
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# %% [markdown]
# # 3. 目標 Recall（例：0.75）を満たす最小の閾値を探す

# %%
desired_recall = 0.75
idx = np.where(recall[:-1] >= desired_recall)[0]
if len(idx) > 0:
    opt_idx = idx[0]
    opt_threshold = thresholds[opt_idx]
    print(f"Recall ≥ {desired_recall:.2f} を満たす最小閾値: {opt_threshold:.3f}")
    print(f"このときの Precision: {precision[opt_idx]:.3f}, Recall: {recall[opt_idx]:.3f}")
else:
    print(f"Recall ≥ {desired_recall:.2f} を満たす閾値は見つかりませんでした。")

# %% [markdown]
# # 4. 新閾値での予測

# %%
y_pred_adj = (y_proba >= opt_threshold).astype(int)
from sklearn.metrics import classification_report, confusion_matrix
print("\n=== 新閾値適用後の分類レポート ===")
print(classification_report(y_test, y_pred_adj, digits=4))
print("=== 新閾値適用後の混同行列 ===")
print(confusion_matrix(y_test, y_pred_adj))

# %% [markdown]
# # ──────────────────────────────
# # ここから F1 スコアを最大化する閾値探索を追加
# # ──────────────────────────────

# %%
from sklearn.metrics import f1_score

# %% [markdown]
# # thresholds 配列に対して、それぞれの閾値での F1 を計算

# %%
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]

# %% [markdown]
# # 最良 F1 を示す閾値を抽出

# %%
best_idx    = int(np.argmax(f1_scores))
best_threshold = thresholds[best_idx]
best_f1     = f1_scores[best_idx]
print(f"\n★ Best F1={best_f1:.3f} を実現する閾値: {best_threshold:.3f}")

# %% [markdown]
# # その閾値で再評価

# %%
y_pred_f1 = (y_proba >= best_threshold).astype(int)
print("\n=== F1最大化閾値適用後の分類レポート ===")
print(classification_report(y_test, y_pred_f1, digits=4))
print("=== F1最大化閾値適用後の混同行列 ===")
print(confusion_matrix(y_test, y_pred_f1))

# %% [markdown]
# # ────────────
# # ① モデルと閾値の保存
# # ────────────

# %%
import joblib
import json
import os

# %% [markdown]
# # モデル保存用ディレクトリを作成

# %%
model_dir = '../models'
os.makedirs(model_dir, exist_ok=True)

# %% [markdown]
# # 1) ランダムフォレストモデルを保存

# %%
model_path = os.path.join(model_dir, 'keiba_rf_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# %% [markdown]
# # 2) 最適閾値と F1 スコアを JSON で保存

# %%
thresh_info = {
    'best_threshold': float(best_threshold),  # F1最大化閾値
    'best_f1_score': float(best_f1)          # そのときの F1
}
thresh_path = os.path.join(model_dir, 'threshold.json')
with open(thresh_path, 'w') as f:
    json.dump(thresh_info, f, indent=2)
print(f"Threshold info saved to {thresh_path}")
