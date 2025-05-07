# scraper/race_result_selenium.py

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def get_race_result(race_id: str) -> pd.DataFrame:
    url = f"https://race.netkeiba.com/race/result.html?race_id={race_id}"

    # ── Chrome オプション設定 ──
    options = Options()
    options.page_load_strategy = "eager"
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.binary_location = "/usr/bin/chromium-browser"

    service = Service("/usr/bin/chromedriver")
    driver  = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)

    try:
        driver.get(url)

        # RaceTable01 または race_table_01 を含む table 要素が見えるまで待機
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((
                By.XPATH,
                "//table[contains(@class,'RaceTable01') or contains(@class,'race_table_01')]"
            ))
        )
        html = driver.page_source

    except Exception as e:
        # タイムアウト時には HTML をファイルに落としてデバッグすると吉
        with open(f"debug_{race_id}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        raise RuntimeError(f"結果テーブルの読み込みに失敗しました: {e}")

    finally:
        driver.quit()

    soup = BeautifulSoup(html, "lxml")
    # BeautifulSoup 側でも同様に複数クラスをフォロー
    table = soup.find("table", class_=["RaceTable01", "race_table_01"])
    if table is None:
        raise RuntimeError(f"BeautifulSoup でテーブルが見つかりませんでした (race_id={race_id})\n"
                           f"-- 保存した debug_{race_id}.html を確認してみてください。")

    rows = table.select("tr")[1:]  # ヘッダー行を除去
    if not rows:
        raise RuntimeError(f"結果行がパースできませんでした (race_id={race_id})")

    data = []
    for row in rows:
        cols   = row.find_all("td")
        rank   = cols[0].get_text(strip=True)
        number = cols[2].get_text(strip=True)
        data.append({
            "rank": int(rank),
            "horse_number": int(number)
        })

    return pd.DataFrame(data)
