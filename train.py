import sys
import pickle
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) != 3:
        print("Usage: python train.py <dataset_location> <model_storage_location>")
        sys.exit(1)
    
    dataset_location = sys.argv[1]
    model_storage_location = sys.argv[2]
    
    # Load dataset
    df = pd.read_csv(dataset_location)
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=5)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Evaluate model
    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)
    print(f"Train R² Score: {train_score:.4f}")
    print(f"Test R² Score: {test_score:.4f}")
    
    # Save model parameters and polynomial configuration
    with open(model_storage_location, 'wb') as f:
        pickle.dump({
            'coef_': model.coef_,
            'intercept_': model.intercept_,
            'poly_degree': 2,
            'poly_features': poly_features
        }, f)
    
    print(f"Model saved to {model_storage_location}")

if __name__ == "__main__":
    main()
