import json

import pandas as pd
from sklearn.model_selection import train_test_split

from metrics_and_plots import plot_learning_curve, save_metrics
from model import evaluate_model, train_model
from utils_and_constants import RAW_DATASET, TARGET_COLUMN
from save_model import save_model


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y


def main():
    X, y = load_data(RAW_DATASET)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    # Train the model using the training set
    model = train_model(X_train, y_train)
    
    # Calculate test set metrics
    metrics = evaluate_model(model, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")
    
    #sauvegarde du model
    save_model(model)

    # Save metrics into json file
    save_metrics(metrics)
    plot_learning_curve(model,X,y)
    


if __name__ == "__main__":
    main()
