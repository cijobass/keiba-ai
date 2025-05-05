# 競馬AIスクレイピングプロジェクト（netkeiba版）

## 概要
netkeiba.comのレース結果を取得してCSVに保存します。

## 構成
- `scraper/`：スクレイピングモジュール
- `scripts/`：バッチ実行用スクリプト
- `data/results/`：取得データの保存先

## 実行方法

```bash
pip install -r requirements.txt
python scripts/fetch_race_results.py