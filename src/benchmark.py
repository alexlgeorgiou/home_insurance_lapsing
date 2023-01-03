import argparse

from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_curve, 
    auc, plot_roc_curve
)
import matplotlib as mpl

import mlflow
import mlflow.sklearn

import pandas as pd 

import pre_process as pp

#TODO: Modularaise main function for maintainability. It's too long, overcoupled...

mpl.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--experiment",
        type=str,
        default='Default',
        help="name of the experiment",
    )
    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    # Extract the data
    df = pd.read_csv('src/data/home_insurance.csv')

    # Clean unviable data and correct types
    clean_df = pp.CleanData(df).clean()
    X_train, X_test, X_val, y_train, y_test, y_val = pp.PreProcess(clean_df).process()

    # enable auto logging
    mlflow.set_experiment(f"{args.experiment}")
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        # train model
        model = DummyClassifier(strategy='stratified')

        model.fit(X_train, y_train)

        # evaluate model metrics
        y_proba = model.predict_proba(X_test)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)

        # log metrics
        mlflow.log_metrics({"auc": auc_score, "log_loss": loss, "accuracy": acc})

        # plot roc_curve
        roc_fig, ax = mpl.pyplot.subplots(1, 1, figsize=(7, 7))
        ax.set_title('ROC Curve')
        plot_roc_curve(model, X_test, y_test, ax=ax)
        mlflow.log_figure(roc_fig, 'roc_curve.png')
        roc_fig.clf()

        # plot probability calibration
        prob_true, prob_pred = calibration_curve(
            y_test, y_proba[:,1], pos_label=1, n_bins=20
        )
        calibration_fig, ax = mpl.pyplot.subplots(1, 1, figsize=(10, 7))
        ax.plot(prob_true, label='true_prob', color='black', linestyle=":")
        ax.plot(prob_pred, label='true_pred', color='blue')
        ax.set_title('Probability Calibration')
        mlflow.log_figure(calibration_fig, 'probability_calibration.png')
        calibration_fig.clf()

if __name__ == "__main__":
    main()