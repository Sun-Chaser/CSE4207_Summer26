import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_input(path: Path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError("Unsupported input file format. Use .csv or .json")


def format_predictions(predictions):
    result = []
    for index, value in enumerate(predictions):
        if hasattr(value, "item"):
            value = value.item()
        result.append({"index": int(index), "prediction": value})
    return result


def get_feature_frame(data, model):
    if isinstance(model, dict) and "feature_names" in model:
        feature_names = model["feature_names"]
        if feature_names and all(name in data.columns for name in feature_names):
            return data.loc[:, feature_names]

    if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
        return data.iloc[:, :-1]

    return data


def predict_with_model(model, data):
    if hasattr(model, "predict"):
        return model.predict(data)

    if callable(model):
        return model(data)

    if isinstance(model, dict):
        if "model" in model and hasattr(model["model"], "predict"):
            return model["model"].predict(get_feature_frame(data, model))

        if {"coef_", "intercept_", "poly_features"}.issubset(model):
            features = get_feature_frame(data, model)
            transformed = model["poly_features"].transform(features)
            return transformed @ model["coef_"] + model["intercept_"]

    raise TypeError("Loaded model does not support prediction")


def main():
    parser = argparse.ArgumentParser(description="Run inference and save predictions to a JSON file.")
    parser.add_argument("model_path", help="Path to the saved model file")
    parser.add_argument("input_path", help="Path to the input dataset file")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_path)

    model = load_model(model_path)
    data = load_input(input_path)

    predictions = predict_with_model(model, data)
    result = format_predictions(predictions)
    output_path = input_path.with_suffix(".predictions.json")

    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=2)

    print(output_path)


if __name__ == "__main__":
    main()
