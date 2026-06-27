import argparse
import pickle
import os
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BUCKET_NAME = os.environ.get("BUCKET_NAME", "cse4207-lab-rebuild")
MODEL_VOLUME_PATH = Path(os.environ.get("MODEL_VOLUME_PATH", "/models"))
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")


s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION
)

def normalize_s3_key(file_key):
    """
    Convert input into a clean S3 object key.

    Accepts:
    - train.csv
    - data/train.csv
    - s3://cse4207-lab-rebuild/data/train.csv
    """
    file_key = str(file_key)

    if file_key.startswith("s3://"):
        path_without_prefix = file_key.replace("s3://", "", 1)
        parts = path_without_prefix.split("/", 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path: {file_key}")

        bucket_name, object_key = parts

        if bucket_name != BUCKET_NAME:
            raise ValueError(
                f"Expected bucket '{BUCKET_NAME}', but got bucket '{bucket_name}'."
            )

        return object_key

    return file_key.lstrip("/")


def read_csv_from_s3(file_key):
    object_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, object_key)

    try:
        body = obj.get()["Body"].read()
    except ClientError as error:
        error_code = error.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            raise FileNotFoundError(
                f"No S3 object found for dataset '{file_key}'. Checked S3 path: "
                f"s3://{BUCKET_NAME}/{object_key}"
            ) from error
        raise

    csv_data = pd.read_csv(BytesIO(body))
    return csv_data


def get_data(dataset_location):
    return read_csv_from_s3(dataset_location)


def write_pickle_to_local_file(file_path, data):
    if str(file_path).startswith("s3://"):
        raise ValueError(
            "Model output must be a local file path on the mounted persistent volume, "
            "not an S3 path."
        )

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = MODEL_VOLUME_PATH / file_path
    elif MODEL_VOLUME_PATH not in (file_path, *file_path.parents):
        raise ValueError(
            f"Model output must be saved under the mounted volume path "
            f"'{MODEL_VOLUME_PATH}'."
        )

    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as model_file:
        pickle.dump(data, model_file)

    return file_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a polynomial regression model and save it to the mounted model volume."
        )
    )

    parser.add_argument(
        "dataset_location",
        help="S3 key or S3 path to the training dataset CSV file.",
    )

    parser.add_argument(
        "model_storage_location",
        help=(
            "Local file path on the mounted persistent volume where the model should "
            "be saved. Relative paths are saved under /models."
        ),
    )

    args = parser.parse_args()

    dataset_location = args.dataset_location
    model_storage_location = args.model_storage_location

    try:
        df = get_data(dataset_location)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if df.shape[1] < 2:
        raise ValueError(
            "Dataset must contain at least one feature column and one target column."
        )

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("Load dataset successfully")

    degree = 5
    poly_features = PolynomialFeatures(degree=degree)

    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    print("Trained model successfully")

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)

    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")

    model_package = {
        "model_type": "polynomial_linear_regression",
        "coef_": model.coef_,
        "intercept_": model.intercept_,
        "poly_degree": degree,
        "poly_features": poly_features,
        "feature_names": feature_names,
        "train_score": train_score,
        "test_score": test_score,
    }

    try:
        saved_path = write_pickle_to_local_file(model_storage_location, model_package)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Model saved to {saved_path}")


if __name__ == "__main__":
    main()
