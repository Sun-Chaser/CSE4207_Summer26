import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    if len(sys.argv) < 2:
        print("Usage: python split.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Load the dataset
    df = pd.read_csv(dataset_path, delimiter=';')
    
    # Split into 80% train and 20% test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save to current directory
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print(f"Train set: {len(train_df)} samples saved to train.csv")
    print(f"Test set: {len(test_df)} samples saved to test.csv")

if __name__ == "__main__":
    main()
