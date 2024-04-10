import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

def build_classifier(model_name, params, random_seed, n_ensemble = 1, verbose = False):

    assert model_name in ["LogisticRegression", "XGBRFClassifier", "XGBRFRegressor", "XGBClassifier", "RidgeClassifier", "RandomForestClassifier", "SVC"], "model_name must be either LogisticRegression, XGBRFClassifier, XGBRFRegressor, XGBClassifier, RidgeClassifier, SVC or RandomForestClassifier"

    classifiers = []
    for idx in range(n_ensemble):
        if model_name == "LogisticRegression":
            if params is None:
                params = {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": random_seed + idx * 100
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = LogisticRegression(**params)
        elif model_name == "XGBRFClassifier":
            if params is None:
                params = {
                    "n_estimators": 20,
                    "max_depth": 1,
                    "random_state": random_seed + idx * 100,
                    "subsample": 0.2,
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = xgb.XGBRFClassifier(**params)
        elif model_name == "XGBRFRegressor":
            if params is None:
                params = {
                    "n_estimators": 200,
                    "max_depth": 1,
                    "random_state": random_seed + idx * 100,
                    "subsample": 0.2,
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = xgb.XGBRFRegressor(**params)
        elif model_name == "XGBClassifier":
            if params is None:
                params = {
                    "n_estimators": 50,
                    "max_depth": 2,
                    "learning_rate": 1.,
                    "verbosity": 0,
                    "random_state": random_seed + idx * 100,
                    "subsample": 0.1,
                }
            classifier = xgb.XGBClassifier(**params)
        elif model_name == "RidgeClassifier":
            if params is None:
                params = {
                    "alpha": 1.0,
                    "fit_intercept": True,
                    "max_iter": None,
                    "tol": 0.001,
                    "class_weight": "balanced",
                    "solver": "auto",
                    "random_state": random_seed + idx * 100,
                    "copy_X": True,
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = RidgeClassifier(**params)  
        if model_name == "RandomForestClassifier":
            if params is None:
                params = {
                    "n_estimators": 50,
                    "criterion": "gini",
                    "max_depth": 1,
                    "min_samples_leaf": 2,
                    "min_weight_fraction_leaf": 0.0,
                    "max_leaf_nodes": None,
                    "random_state": random_seed + idx * 100,
                    "class_weight": "balanced",
                    "max_features": "log2"
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = RandomForestClassifier(**params)
        if model_name == "SVC":
            if params is None:
                params = {
                    "C": 1.0,
                    "kernel": "rbf",
                    "degree": 3,
                    "gamma": "scale",
                    "coef0": 0.0,
                    "shrinking": True,
                    "probability": False,
                    "cache_size": 200,
                    "class_weight": "balanced",
                    "decision_function_shape": "ovr",
                    "random_state": random_seed + idx * 100,
                    "probability": True
                }
            params["random_state"] = params["random_state"] + idx * 100
            classifier = SVC(**params)
        classifiers.append(classifier)
    
    params["random_state"] = random_seed

    if verbose:
        print(f"Building {model_name} classifier with following parameters:")
        for key, value in params.items():
            print(f"\t{key}: {value}")

    return classifiers, params

def evaluate_model(model_name, results_summary, plot=True, output_dir=None, verbose=True, optimal_threshold_criterion="youden"):
    if "y_class_train" in results_summary:
        y_train = results_summary["y_class_train"]
        y_test = results_summary["y_class_test"]
    else:
        y_train = results_summary["y_train"]
        y_test = results_summary["y_test"]

    # Read regression prediction from summary
    probs_train = results_summary['probs_train']
    probs_test = results_summary['probs_test']

    # Compute tpr, fpr, roc_auc from train predictions
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, probs_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    results_summary['fpr_train'] = fpr_train
    results_summary['tpr_train'] = tpr_train
    results_summary['roc_auc_train'] = roc_auc_train

    # Compute non-weighted f1 score to determine optimal threshold
    precision_train, recall_train, thresholds_train = precision_recall_curve(y_train, probs_train)
    f1_scores_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)
    # Substitute nans for 0
    f1_scores_train = np.nan_to_num(f1_scores_train)
    optimal_threshold_train = thresholds_train[np.argmax(tpr_train - fpr_train)]
    results_summary['optimal_threshold_train'] = optimal_threshold_train
    # Compute class predictions and weighted f1 score with optimal threshold
    y_train_pred = np.where(probs_train >= optimal_threshold_train, 1, 0)
    results_summary['y_train_pred'] = y_train_pred
    f1_score_train =  f1_score(y_train, y_train_pred, average="weighted")
    results_summary['f1_score_train'] = f1_score_train

    # Compute tpr, fpr, roc_auc from test predictions
    fpr_test, tpr_test, thresholds_test_roc = roc_curve(y_test, probs_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    results_summary['fpr_test'] = fpr_test
    results_summary['tpr_test'] = tpr_test
    results_summary['roc_auc_test'] = roc_auc_test

    # Compute non-weighted f1 score to determine optimal threshold
    precision_test, recall_test, thresholds_test_pr = precision_recall_curve(y_test, probs_test)
    precision_test[0] = 1
    precision_test[-1] = 0
    recall_test[0] = 0
    recall_test[-1] = 1

    if optimal_threshold_criterion == "youden":
        # Compute optimal threshold by maximizing the Youden's index
        youden_index = tpr_test - fpr_test
        optimal_threshold_test = thresholds_test_roc[np.argmax(youden_index)]
        y_test_pred = (probs_test > optimal_threshold_test).astype(int)
    elif optimal_threshold_criterion == "f1":
        # Compute optimal threshold by maximizing the F1-score
        f1_scores_ = 2 * (precision_test * recall_test) / (precision_test + recall_test)
        f1_scores_ = np.nan_to_num(f1_scores_)
        optimal_threshold_test = thresholds_test_pr[np.argmax(f1_scores_)]
        y_test_pred = (probs_test > optimal_threshold_test).astype(int)
    elif optimal_threshold_criterion == "f15":
        # Compute optimal threshold by maximizing the F1.5-score
        f15_scores_ = (1 + 1.5 ** 2) * (precision_test * recall_test) / (1.5 ** 2 * precision_test + recall_test)
        f15_scores_ = np.nan_to_num(f15_scores_)
        optimal_threshold_test = thresholds_test_pr[np.argmax(f15_scores_)]
        y_test_pred = (probs_test > optimal_threshold_test).astype(int)
    elif optimal_threshold_criterion == "f2":
        # Compute optimal threshold by maximizing the F2-score
        f2_scores_ = (1 + 2 ** 2) * (precision_test * recall_test) / (2 ** 2 * precision_test + recall_test)
        f2_scores_ = np.nan_to_num(f2_scores_)
        optimal_threshold_test = thresholds_test_pr[np.argmax(f2_scores_)]
        y_test_pred = (probs_test > optimal_threshold_test).astype(int)
    elif optimal_threshold_criterion == "f05":
        # Compute optimal threshold by maximizing the F0.5-score
        f05_scores_ = (1 + 0.5 ** 2) * (precision_test * recall_test) / (0.5 ** 2 * precision_test + recall_test)
        f05_scores_ = np.nan_to_num(f05_scores_)
        optimal_threshold_test = thresholds_test_pr[np.argmax(f05_scores_)]
        y_test_pred = (probs_test > optimal_threshold_test).astype(int)
        
    results_summary['optimal_threshold_test'] = optimal_threshold_test

    # Compute class predictions and weighted f1 score with optimal threshold (test)
    y_test_pred = np.where(probs_test > optimal_threshold_test, 1, 0)
    results_summary['y_test_pred'] = y_test_pred
    f1_score_test =  f1_score(y_test, y_test_pred, average="weighted")
    results_summary['f1_score_test'] = f1_score_test

    # Compute class predictions and weighted f1 score with optimal threshold (train)
    y_test_train_pred = np.where(probs_test >= optimal_threshold_train, 1, 0)
    results_summary['y_test_train_class_pred'] = y_test_train_pred
    f1_score_test_train =  f1_score(y_test, y_test_train_pred, average="weighted")
    results_summary['f1_score_test_train'] = f1_score_test_train

    if verbose:
        print("Results for {}".format(model_name))
        print("Train: f1_score = {}, roc_auc = {}".format(round(f1_score_train, 2), round(roc_auc_train, 2)))
        print("Test: f1_score = {}, roc_auc = {}".format(round(f1_score_test_train, 2), round(roc_auc_test, 2)))

    if plot:
        make_prob_dist_plots(results_summary['model_name'], y_train, probs_train, y_test, probs_test, optimal_threshold_train, optimal_threshold_test, output_dir)
        plot_roc(results_summary['model_name'], fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, output_dir)

    return results_summary

def make_prob_dist_plots(model_name, y_train, probs_train, y_test, probs_test, optimal_threshold, optimal_threshold_test, output_dir = None):
    # Make dataframes with true values and predictions
    df_train = pd.DataFrame({"classification": y_train, "predictions": probs_train})
    df_test = pd.DataFrame({"classification": y_test, "predictions": probs_test})

    # According to predictions column and classification column, create a new column for tp, fp, tn, fn
    df_train['tp'] = np.nan
    df_train['fp'] = np.nan
    df_train['tn'] = np.nan
    df_train['fn'] = np.nan
    # Classify predictions
    df_train['tp'] = np.where((df_train['classification'].isin([1, 2])) & (df_train['predictions'] >= optimal_threshold), df_train['predictions'], np.nan)
    df_train['tn'] = np.where((df_train['classification'] == 0) & (df_train['predictions'] < optimal_threshold), df_train['predictions'], np.nan)
    df_train['fp'] = np.where((df_train['classification'] == 0) & (df_train['predictions'] >= optimal_threshold), df_train['predictions'], np.nan)
    df_train['fn'] = np.where((df_train['classification'].isin([1, 2])) & (df_train['predictions'] < optimal_threshold), df_train['predictions'], np.nan)

    # According to predictions column and classification column, create a new column for tp, fp, tn, fn
    df_test['tp'] = np.nan
    df_test['fp'] = np.nan
    df_test['tn'] = np.nan
    df_test['fn'] = np.nan
    # Classify predictions
    df_test['tp'] = np.where((df_test['classification'].isin([1, 2])) & (df_test['predictions'] >= optimal_threshold_test), df_test['predictions'], np.nan)
    df_test['tn'] = np.where((df_test['classification'] == 0) & (df_test['predictions'] < optimal_threshold_test), df_test['predictions'], np.nan)
    df_test['fp'] = np.where((df_test['classification'] == 0) & (df_test['predictions'] >= optimal_threshold_test), df_test['predictions'], np.nan)
    df_test['fn'] = np.where((df_test['classification'].isin([1, 2])) & (df_test['predictions'] < optimal_threshold_test), df_test['predictions'], np.nan)

    # Make a subplot to plot the data side by side
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(df_train['tp'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='TP')
    ax[0].hist(df_train['fp'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='FP')
    ax[0].hist(df_train['tn'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='TN')
    ax[0].hist(df_train['fn'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='FN')
    ax[0].axvline(optimal_threshold, color='red')
    ax[0].set_title('Train')
    ax[0].set_xlabel('Prediction')
    ax[0].set_ylabel('Counts')
    ax[0].set_xlim([0, 1])
    ax[0].legend(loc='upper right')
    ax[1].hist(df_test['tp'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='TP')
    ax[1].hist(df_test['fp'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='FP')
    ax[1].hist(df_test['tn'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='TN')
    ax[1].hist(df_test['fn'].dropna(), bins=20, range=(0, 1), alpha=0.5, label='FN')
    ax[1].axvline(optimal_threshold_test, color='red')
    ax[1].set_title('Test')
    ax[1].set_xlabel('Prediction')
    ax[1].set_ylabel('Counts')
    ax[1].set_xlim([0, 1])
    ax[1].legend(loc='upper right')

    # Set title for the whole plot
    plt.suptitle("Probability distributions for {}".format(model_name))

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "prob_dist.jpeg"))
        plt.close()

def plot_roc(model_name, fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test, output_dir = None):
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(fpr_train, tpr_train, color='red', lw=2, label='ROC curve (area = {})'.format(round(roc_auc_train, 2)))
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve (train)")
    ax[0].legend(loc="lower right")

    ax[1].plot(fpr_test, tpr_test, color='red', lw=2, label='ROC curve (area = {})'.format(round(roc_auc_test, 2)))
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlabel("False positive rate")
    ax[1].set_ylabel("True positive rate")
    ax[1].set_title("ROC curve (test)")
    ax[1].legend(loc="lower right")

    # Set title for the whole plot
    plt.suptitle("ROC for {}".format(model_name))

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "roc.jpeg"))
        plt.close()