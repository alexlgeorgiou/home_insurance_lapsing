import argparse
import sys

import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, log_loss, roc_curve, 
    auc, plot_roc_curve
)
import xgboost as xgb
import matplotlib as mpl


import mlflow
import mlflow.xgboost

import pandas as pd 

import pre_process as pp

#TODO: Modularaise main function for maintainability. It's too long, overcoupled...
#TODO: modify for 'training performance' deploy decision. i.e. Is this model better than last?

mpl.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost example")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=10,
        help="The number of estimators to use. (default 10)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="number of levels each tree go extend to (default 10)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default='Default',
        help="name of the experiment",
    )
    parser.add_argument(
        "--final-features",
        action='store_true',
        help="flag to enable the filtered features",
    )
    return parser.parse_args()


def main():
    #TODO: This main function is way to long, coupled and not built as a interface. Needs refactoring. 
    # parse command-line arguments
    args = parse_args()

    # prepare train and test data
    # Extract the data
    df = pd.read_csv('src/data/home_insurance.csv')

    # Clean unviable data and correct types
    clean_df = pp.CleanData(df).clean()
    
    # Split out data
    X_train, X_test, X_val, y_train, y_test, y_val = (
        pp.PreProcess(clean_df)
            .process(important_features=args.final_features)
    )

    # enable auto logging
    mlflow.set_experiment(f"/{args.experiment}")
    mlflow.xgboost.autolog()

    with mlflow.start_run():

        # train model
        model = xgb.XGBClassifier(
            n_estimators=10, 
            max_depth=args.max_depth,
            colsample_bytree=args.colsample_bytree,
            subsample=args.subsample,
            n_jobs=-2, 
            learning_rate=args.learning_rate, 
            objective='binary:logistic'
        )

        model.fit(X_train, y_train)

        # evaluate model metrics
        y_proba = model.predict_proba(X_val)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_val, y_proba)
        acc = accuracy_score(y_val, y_pred)
        fpr, tpr, thresholds = roc_curve(y_val, y_proba[:, 1], pos_label=1)
        auc_score = auc(fpr, tpr)

        # log metrics
        mlflow.log_metrics({"auc": auc_score, "log_loss": loss, "accuracy": acc})
        
        # plot probability calibration
        prob_true, prob_pred = calibration_curve(
            y_val, y_proba[:,1], pos_label=1, n_bins=20
        )
        calibration_fig, ax = mpl.pyplot.subplots(1, 1, figsize=(10, 7))
        ax.plot(prob_true, label='true_prob', color='black', linestyle=":")
        ax.plot(prob_pred, label='true_pred', color='blue')
        ax.set_title('Probability Calibration')
        mlflow.log_figure(calibration_fig, 'probability_calibration.png')
        calibration_fig.clf()
        
        # run shapley values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)

        # plot roc_curve
        roc_fig, ax = mpl.pyplot.subplots(1, 1, figsize=(7, 7))
        ax.set_title('ROC Curve')
        plot_roc_curve(model, X_val, y_val, ax=ax)
        mlflow.log_figure(roc_fig, 'roc_curve.png')

        roc_fig.clf() 

        # plot shap summary
        shap.summary_plot(shap_values, X_val, show=False)
        shap_fig = mpl.pyplot.gcf()
        mlflow.log_figure(shap_fig, 'shap_summary.png')

        shap_fig.clf()

        for col_name in X_val.columns:
            try:
                n_vals = X_val[col_name].nunique()
                apply_jitter = lambda x: 0.6 if x < 30 else 0.0
                (shap
                    .dependence_plot(
                        col_name, shap_values, 
                        X_val, 
                        interaction_index=None,
                        x_jitter=apply_jitter(n_vals),
                        alpha=0.8,
                        show=False)
                )
                shap_feature_fig = mpl.pyplot.gcf()
                mlflow.log_figure(shap_feature_fig, f'{col_name}.png')
                shap_feature_fig.clf()
            except Exception as e:
                print(f'Error plotting dependance plots: {col_name}')
                print(e)
                sys.exit()


if __name__ == "__main__":
    main()