# Import libraries

import argparse
import glob
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



# define functions
def main(args):
    # TO DO: enable autologging
    # Start Logging
    mlflow.start_run()

    # enable autologging
    # mlflow.sklearn.autolog()
    mlflow.autolog()

    # print training run options and hyperparameters to the console
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    # read data
    df = get_csvs_df(args.training_data)

    mlflow.log_metric("num_samples", df.shape[0])
    mlflow.log_metric("num_features", df.shape[1] - 1)

    # split data
    X_train, X_test, y_train, y_test = split_data(df, args.test_train_ratio)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    # evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print(" ".join(f"{k}={v}" for k, v in metrics.items()))


    # Registering the model to the workspace
    # print("Registering the model via MLFlow")
    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     registered_model_name=args.registered_model_name,
    #     artifact_path=args.registered_model_name,
    # )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=model,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )

    # Stop Logging
    mlflow.end_run()

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df, test_train_ratio):
    X, y = df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
               'DiabetesPedigree', 'Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_ratio, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])

    # # plot ROC curve
    # fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    # fig = plt.figure(figsize=(6, 4))
    # # Plot the diagonal 50% line
    # plt.plot([0, 1], [0, 1], 'k--')
    # # Plot the FPR and TPR achieved by our model
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    metrics = {
        "Accuracy": acc,
        "ROC AUC": auc
    }
    return metrics


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.3)
    parser.add_argument("--registered_model_name", type=str, required=Fale, default="TheAwesomeModel",help="model name")


    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
