import argparse
import json
import pickle
from io import BytesIO
import os

import boto3
import pandas as pd


BUCKET_NAME = os.environ.get("BUCKET_NAME", "cse4207-lab-rebuild")


s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.environ.get("PERSONAL_ACCESS_TOKEN"),
    aws_secret_access_key=os.environ.get("SUPER_SECRET_PASSWORD"),
)


def normalize_s3_key(file_key):
    """
    Convert user input into a clean S3 object key.

    Accepts:
    - train.csv
    - folder/train.csv
    - s3://bucket-name/folder/train.csv
    """
    file_key = str(file_key)

    if file_key.startswith("s3://"):
        parts = file_key.replace("s3://", "", 1).split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {file_key}")

        bucket, key = parts
        if bucket != BUCKET_NAME:
            raise ValueError(
                f"This script is configured for bucket '{BUCKET_NAME}', "
                f"but received path from bucket '{bucket}'."
            )
        return key

    return file_key.lstrip("/")


def load_model(file_key):
    file_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, file_key)
    body = obj.get()["Body"].read()

    model = pickle.load(BytesIO(body))
    return model


def load_input(file_key):
    file_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, file_key)
    body = obj.get()["Body"].read()

    csv_data = pd.read_csv(BytesIO(body))
    return csv_data


def write_data(file_key, save_data):
    file_key = normalize_s3_key(file_key)

    if not isinstance(save_data, str):
        save_data = json.dumps(save_data, indent=2)

    obj = s3.Object(BUCKET_NAME, file_key)
    obj.put(
        Body=save_data.encode("utf-8"),
        ContentType="application/json",
    )

    return file_key


def format_predictions(predictions):
    result = []

    for index, value in enumerate(predictions):
        if hasattr(value, "item"):
            value = value.item()

        result.append(
            {
                "index": int(index),
                "prediction": value,
            }
        )

    return result


def get_feature_frame(data, model=None):
    """
    Select only feature columns for prediction.

    Priority:
    1. Use feature_names saved in model dictionary.
    2. Use sklearn model.feature_names_in_ if available.
    3. If dataframe has more than one column, assume the last column may be label.
    4. Otherwise use the full dataframe.
    """

    if isinstance(model, dict) and "feature_names" in model:
        feature_names = model["feature_names"]
        if feature_names and all(name in data.columns for name in feature_names):
            return data.loc[:, feature_names]

    sklearn_model = None

    if isinstance(model, dict) and "model" in model:
        sklearn_model = model["model"]
    else:
        sklearn_model = model

    if hasattr(sklearn_model, "feature_names_in_"):
        feature_names = list(sklearn_model.feature_names_in_)
        if all(name in data.columns for name in feature_names):
            return data.loc[:, feature_names]

    if isinstance(data, pd.DataFrame) and len(data.columns) > 1:
        return data.iloc[:, :-1]

    return data


def predict_with_model(model, data):
    if isinstance(model, dict):
        if "model" in model and hasattr(model["model"], "predict"):
            features = get_feature_frame(data, model)
            return model["model"].predict(features)

        if {"coef_", "intercept_", "poly_features"}.issubset(model.keys()):
            features = get_feature_frame(data, model)
            transformed = model["poly_features"].transform(features)
            return transformed @ model["coef_"] + model["intercept_"]

    if hasattr(model, "predict"):
        features = get_feature_frame(data, model)
        return model.predict(features)

    if callable(model):
        return model(data)

    raise TypeError("Loaded model does not support prediction.")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference and save predictions to an S3 JSON file."
    )

    parser.add_argument("model_path", help="S3 key or S3 path to the saved model file")
    parser.add_argument("input_path", help="S3 key or S3 path to the input dataset file")
    parser.add_argument("save_path", help="S3 key or S3 path to save predictions")

    args = parser.parse_args()

    model = load_model(args.model_path)
    data = load_input(args.input_path)

    predictions = predict_with_model(model, data)
    result = format_predictions(predictions)

    write_data(args.save_path, result)

    print(f"Prediction results have been saved to s3://{BUCKET_NAME}/{normalize_s3_key(args.save_path)}")


if __name__ == "__main__":
    main()