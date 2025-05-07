import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_race_entry(race_id: str) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    res = requests.get(url, headers=headers)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "lxml")

    table = soup.find("table", class_="Shutuba_Table")
    if table is None:
        raise RuntimeError(f"出馬表が見つかりません: race_id={race_id}")

    data = []
    # 最初のヘッダー行(s)をスキップ
    for row in table.find_all("tr")[2:]:
        cols = row.find_all("td")
        # 枠番・馬番
        waku = cols[0].get_text(strip=True)
        uma  = cols[1].get_text(strip=True)

        # ————— ここを修正 —————
        weight_text = cols[4].get_text(strip=True)  # 斤量
        try:
            weight_val = float(weight_text)
        except (ValueError, TypeError):
            weight_val = None
        # ——————————————

        jockey  = cols[5].get_text(strip=True)      # 騎手
        trainer = cols[6].get_text(strip=True)      # 調教師

        body = cols[7].get_text(strip=True)         # 馬体重＋増減
        declared = body.split("(")[0] if body else ""

        data.append({
            "枠番":              int(waku) if waku.isdigit() else waku,
            "馬番":              int(uma)  if uma.isdigit()  else uma,
            "斤量":              weight_val,
            "馬体重":            int(declared) if declared.isdigit() else declared,
            "騎手":              jockey,
            "調教師":            trainer,
            "単勝オッズ":        None,   # あとで埋めます
        })

    df = pd.DataFrame(data)

    # ——— 単勝オッズ取得部（省略せずそのまま） ———
    odds_url = f"https://race.netkeiba.com/race/odds.html?race_id={race_id}"
    res2 = requests.get(odds_url, headers=headers)
    res2.encoding = res2.apparent_encoding
    soup2 = BeautifulSoup(res2.text, "lxml")
    odds_table = (
        soup2.find("table", class_="Odds_Table")
        or soup2.find("table", class_="odds_table_01")
    )
    if odds_table:
        for row in odds_table.find_all("tr")[1:]:
            cols2 = row.find_all("td")
            if len(cols2) >= 2:
                horse_no = cols2[0].get_text(strip=True)
                odd      = cols2[1].get_text(strip=True)
                try:
                    horse_no_i = int(horse_no)
                    odd_f      = float(odd)
                    df.loc[df["馬番"] == horse_no_i, "単勝オッズ"] = odd_f
                except ValueError:
                    continue

    df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    return df
