# 🏇 Keiba AI - 競馬レース予測モデル

このプロジェクトは、過去の競馬レースデータを用いて、出走馬の勝利予測を行う機械学習モデルの構築を目的としています。

---

## 📁 ディレクトリ構成

keiba-ai/
├── data/
│ ├── raw/ # 元データ（race-result-horse.csvなど）
│ ├── processed/ # 前処理済みデータ（processed_data.csv）
│ └── results/ # 予測結果ファイル（predictions.csv）
├── models/ # 学習済みモデル & 閾値ファイル
│ ├── keiba_rf_model.pkl
│ ├── keiba_rf_model_balanced.pkl
│ ├── keiba_rf_model_smote.pkl
│ ├── keiba_rf_model_tuned.pkl
│ └── threshold.json
├── notebooks/
│ ├── model_building.ipynb # 実験・検証用ノートブック
│ └── model_building.py
├── results/ # ※重複注意（data/results/と統合推奨）
├── scripts/ # 各種スクリプト
│ ├── preprocess.py # データ前処理
│ ├── train.py # モデル学習（バランス/SMOTE/チューニング対応）
│ ├── tune_threshold.py # 最適なしきい値を探索・保存
│ ├── predict.py # 学習済みモデルでの予測
│ ├── check_model_features.py # モデルとデータの特徴量整合性確認
│ └── fetch_race_results.py # レース結果取得（スクレイピング）
├── scraper/
│ └── race_result_scraper.py # スクレイピングモジュール
├── .python-version # pyenv用
├── requirements.txt # 依存パッケージ一覧
└── README.md # このファイル


---

## ✅ 実行手順

### 1. データ前処理

```bash
python scripts/preprocess.py

2. モデル学習（例：チューニング版）
bash
コピーする
編集する
python scripts/train.py
3. 閾値最適化
bash
コピーする
編集する
python scripts/tune_threshold.py \
  --model models/keiba_rf_model_tuned.pkl \
  --data data/processed/processed_data.csv \
  --output models/threshold.json
4. 予測実行
bash
コピーする
編集する
python scripts/predict.py \
  --input data/processed/processed_data.csv \
  --output data/results/predictions.csv \
  --model models/keiba_rf_model_tuned.pkl \
  --threshold models/threshold.json
🧠 使用技術・ライブラリ
Python 3.11

scikit-learn

imbalanced-learn

pandas, numpy

joblib

matplotlib / seaborn（Notebook用）

📈 モデルバリエーション
モデル名	特徴
keiba_rf_model.pkl	ベースライン（未調整）
keiba_rf_model_balanced.pkl	class_weight='balanced' 利用
keiba_rf_model_smote.pkl	SMOTEで少数クラス補正
keiba_rf_model_tuned.pkl	GridSearchCVでチューニング済み

✍️ 補足
閾値ファイル（threshold.json）は best_threshold or threshold キーでしきい値を保持します。

Notebook はプロトタイプ実験用途、scripts/ 配下はバッチ・本番処理想定です。

🗂 今後の展望（TODO）
 外部データソース（血統・コース情報）との統合

 モデルの軽量化とリアルタイム適用

 可視化ダッシュボードの実装

📄 ライセンス
このプロジェクトは個人の学習・研究用途を想定しており、特別なライセンスは設定していません。


---
