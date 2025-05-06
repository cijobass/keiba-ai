import joblib
import pandas as pd
import argparse

def main(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    model_features = model.named_steps['rf'].feature_names_in_
    data_features = df.drop('target', axis=1, errors='ignore').columns.to_list()

    print("=== モデルが期待する特徴量 ===")
    print(model_features)

    print("\n=== 実際のデータの特徴量 ===")
    print(data_features)

    # 特徴量の差を表示
    missing_in_data = set(model_features) - set(data_features)
    extra_in_data = set(data_features) - set(model_features)

    print("\n✅ モデルに存在して、データに存在しない特徴量:")
    print(missing_in_data)

    print("\n⚠️ データに存在して、モデルに存在しない特徴量:")
    print(extra_in_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    main(args.model, args.data)
