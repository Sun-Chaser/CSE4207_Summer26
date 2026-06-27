import argparse
import json
import pickle
from io import BytesIO
import os
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError


BUCKET_NAME = os.environ.get("BUCKET_NAME", "cse4207-lab-rebuild")
MODEL_VOLUME_PATH = Path(os.environ.get("MODEL_VOLUME_PATH", "/models"))


s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
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


def resolve_volume_path(file_path, path_type):
    if str(file_path).startswith("s3://"):
        raise ValueError(
            f"{path_type} must be a local file path on the mounted persistent volume, "
            "not an S3 path."
        )

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = MODEL_VOLUME_PATH / file_path
    elif MODEL_VOLUME_PATH not in (file_path, *file_path.parents):
        raise ValueError(
            f"{path_type} must be under the mounted volume path '{MODEL_VOLUME_PATH}'."
        )

    return file_path


def load_model(file_path):
    file_path = resolve_volume_path(file_path, "Model path")

    with file_path.open("rb") as model_file:
        model = pickle.load(model_file)

    return model


def load_input(file_key):
    object_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, object_key)

    try:
        body = obj.get()["Body"].read()
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(
                f"No S3 object found for input dataset '{file_key}'. Checked S3 path: "
                f"s3://{BUCKET_NAME}/{object_key}"
            ) from error
        raise

    csv_data = pd.read_csv(BytesIO(body))
    return csv_data


def write_data(file_path, save_data):
    if not isinstance(save_data, str):
        save_data = json.dumps(save_data, indent=2)

    file_path = resolve_volume_path(file_path, "Prediction output path")

    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as output_file:
        output_file.write(save_data)

    return file_path


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
        description=(
            "Run inference using a model from the mounted volume and save predictions "
            "to the mounted volume."
        )
    )

    parser.add_argument(
        "model_path",
        help="Local path on the mounted persistent volume to the saved model file.",
    )
    parser.add_argument("input_path", help="S3 key or S3 path to the input dataset file")
    parser.add_argument(
        "save_path",
        help=(
            "Local path on the mounted persistent volume to save predictions. "
            "Relative paths are saved under /models."
        ),
    )

    args = parser.parse_args()

    try:
        model = load_model(args.model_path)
        data = load_input(args.input_path)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    predictions = predict_with_model(model, data)
    result = format_predictions(predictions)

    try:
        saved_path = write_data(args.save_path, result)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Prediction results have been saved to {saved_path}")


if __name__ == "__main__":
    main()
