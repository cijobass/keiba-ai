FROM python:3.11-slim

# 1) 必要パッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    ca-certificates \
    # Chromium 本体
    chromium \
    # Chromium 用 WebDriver
    chromium-driver \
  && rm -rf /var/lib/apt/lists/*

# 2) 環境変数でパスを通しておく
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# 3) 作業ディレクトリ設定
WORKDIR /app

# 4) 依存ライブラリをコピー＆インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) ソースをコピー
COPY . .

# 6) デフォルトコマンド
CMD ["bash"]
