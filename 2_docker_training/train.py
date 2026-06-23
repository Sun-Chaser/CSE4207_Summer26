import argparse
import pickle
import os
from io import BytesIO

import boto3
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BUCKET_NAME = os.environ.get("BUCKET_NAME", "cse4207-lab-rebuild")

s3 = boto3.resource(
    "s3",
    aws_access_key_id=os.environ.get("PERSONAL_ACCESS_TOKEN"),
    aws_secret_access_key=os.environ.get("SUPER_SECRET_PASSWORD"),
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


def get_data(file_key):
    file_key = normalize_s3_key(file_key)

    obj = s3.Object(BUCKET_NAME, file_key)
    body = obj.get()["Body"].read()

    csv_data = pd.read_csv(BytesIO(body))
    return csv_data


def write_pickle_to_s3(file_key, data):
    file_key = normalize_s3_key(file_key)

    pickled_data = pickle.dumps(data)

    obj = s3.Object(BUCKET_NAME, file_key)
    obj.put(Body=pickled_data)

    return file_key


def main():
    parser = argparse.ArgumentParser(
        description="Train a polynomial regression model and save it to S3."
    )

    parser.add_argument(
        "dataset_location",
        help="S3 key or S3 path to the training dataset CSV file.",
    )

    parser.add_argument(
        "model_storage_location",
        help="S3 key or S3 path where the trained model should be saved.",
    )

    args = parser.parse_args()

    dataset_location = args.dataset_location
    model_storage_location = args.model_storage_location

    df = get_data(dataset_location)

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

    degree = 5
    poly_features = PolynomialFeatures(degree=degree)

    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

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

    saved_key = write_pickle_to_s3(model_storage_location, model_package)

    print(f"Model saved to s3://{BUCKET_NAME}/{saved_key}")


if __name__ == "__main__":
    main()