# scraper/race_result_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_race_result(race_id: str) -> pd.DataFrame:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    res = requests.get(url, headers=headers)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "lxml")
    
    print(soup.prettify()[:2000])  # 最初の2000文字だけ表示

    rows = soup.select("table.race_table_01 tr")[1:]  # ヘッダー除く
    result = []

    for row in rows:
        cols = row.find_all("td")
        print(f"Row length: {len(cols)}")
        if len(cols) < 8:  # ← 最低限の列数だけ確認
            continue
        result.append({
            "race_id": race_id,
            "rank": cols[0].get_text(strip=True),
            "frame": cols[1].get_text(strip=True),
            "number": cols[2].get_text(strip=True),
            "horse": cols[3].get_text(strip=True),
            "sex_age": cols[4].get_text(strip=True),
            "jockey": cols[6].get_text(strip=True),
            "time": cols[7].get_text(strip=True),
            # 念のための補完
            "odds": cols[10].get_text(strip=True) if len(cols) >= 11 else "",
        })

    return pd.DataFrame(result)
