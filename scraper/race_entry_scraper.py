import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_race_entry(race_id: str) -> pd.DataFrame:
    """
    race_id の出馬表をスクレイピングし、
    馬番 (horse_number) をインデックスにした DataFrame を返します。
    カラム: frame, horse_number, weight, declared_weight, jockey, trainer, win_odds
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    res = requests.get(url, headers=headers)
    res.encoding = res.apparent_encoding
    soup = BeautifulSoup(res.text, "lxml")

    table = soup.find("table", class_="Shutuba_Table")
    if table is None:
        raise RuntimeError(f"出馬表が見つかりません: race_id={race_id}")

    data = []
    # ヘッダー行をスキップして出走馬ごとにループ
    for row in table.find_all("tr")[2:]:
        cols = row.find_all("td")
        waku    = cols[0].get_text(strip=True)     # 枠番
        uma     = cols[1].get_text(strip=True)     # 馬番

        # 斤量
        weight_text = cols[4].get_text(strip=True)
        try:
            weight_val = float(weight_text)
        except (ValueError, TypeError):
            weight_val = None

        jockey  = cols[5].get_text(strip=True)     # 騎手
        trainer = cols[6].get_text(strip=True)     # 調教師

        # 馬体重（例: "500(+2)" の "500" 部分だけ）
        body     = cols[7].get_text(strip=True)
        declared = body.split("(")[0] if body else ""
        try:
            declared_val = int(declared)
        except (ValueError, TypeError):
            declared_val = None

        data.append({
            "frame":            int(waku) if waku.isdigit() else waku,
            "horse_number":     int(uma)  if uma.isdigit()  else uma,
            "weight":           weight_val,
            "declared_weight":  declared_val,
            "jockey":           jockey,
            "trainer":          trainer,
            "win_odds":         None,   # 後で埋める
        })

    df = pd.DataFrame(data)

    # ――― 単勝オッズを取得して win_odds に埋める ―――
    odds_url = f"https://race.netkeiba.com/race/odds.html?race_id={race_id}"
    res2     = requests.get(odds_url, headers=headers)
    res2.encoding = res2.apparent_encoding
    soup2    = BeautifulSoup(res2.text, "lxml")

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
                    df.loc[df["horse_number"] == horse_no_i, "win_odds"] = odd_f
                except ValueError:
                    continue

    # 数値化しておく
    df["win_odds"] = pd.to_numeric(df["win_odds"], errors="coerce")

    # horse_number をインデックスにして返す
    return df.set_index("horse_number")
