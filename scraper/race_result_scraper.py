import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_race_result(race_id: str) -> pd.DataFrame:
    """
    race_id のレース結果をスクレイピングし、
    必須列 'horse_number'（馬番）, 'rank'（着順）を返します。
    """

    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "lxml")

    # 1) 候補となる result テーブルを探す
    tables = soup.find_all("table")
    result_table = None
    for t in tables:
        cls = t.get("class") or []
        # class に "race_table" を含むものを結果テーブルとみなす
        if any("race_table" in c for c in cls):
            result_table = t
            break

    if result_table is None:
        raise RuntimeError(f"結果テーブルが見つかりません (race_id={race_id})")

    # 2) <tr> をすべて取得し、ヘッダー行をスキップ
    rows = result_table.find_all("tr")[1:]
    data = []
    for row in rows:
        cols = row.find_all("td")
        # 着順・馬番を含む最低限の列数かチェック
        if len(cols) < 3:
            continue
        # 着順取得
        txt0 = cols[0].get_text(strip=True)
        txt2 = cols[2].get_text(strip=True)
        try:
            rank = int(txt0)
            horse_number = int(txt2)
        except ValueError:
            continue
        data.append({
            "race_id":     race_id,
            "horse_number": horse_number,
            "rank":         rank,
        })

    if not data:
        raise RuntimeError(f"パースできた結果行がありません (race_id={race_id})")

    df = pd.DataFrame(data)
    df["horse_number"] = df["horse_number"].astype(int)
    df["rank"]         = df["rank"].astype(int)
    return df
