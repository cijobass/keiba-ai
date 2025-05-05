# scripts/fetch_race_results.py

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scraper.race_result_scraper import get_race_result

# 取得したいレースIDをリストで指定
race_ids = [
    "202405030811",  # 天皇賞・春
    "202405030712",  # 他のレース（例）
]

output_dir = "data/results"
os.makedirs(output_dir, exist_ok=True)

for race_id in race_ids:
    print(f"Fetching {race_id} ...")
    df = get_race_result(race_id)
    output_path = os.path.join(output_dir, f"{race_id}.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {output_path}")
