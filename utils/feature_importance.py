import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def rfe_plot(rfe_df, output_dir=None):
    """
    Makes recursive feature elimination (RFE) plot. This plot will show the evolution of the AUC as a 
    function of the number of features.

    Parameters
    ----------
    rfe_df : pd.DataFrame
        A dataframe containing the results of the RFE. It should contain the following columns:
        - "N_features": the number of features
        - "Train ROC": the mean test score
        - "Train ROC (std)": the standard deviation of the test score
        - "Test ROC": the mean train score
        - "Test ROC (std)": the standard deviation of the train score

    """
    _, ax = plt.subplots(figsize=(7, 5), dpi=1000)
    ax.grid(lw=0.5, alpha=0.5)
    ax.errorbar(rfe_df["N_features"], rfe_df["Train ROC"], yerr=rfe_df["Train ROC (std)"], fmt='-o', label="Train ROC", color = "mediumblue", lw = 1.25)
    ax.errorbar(rfe_df["N_features"], rfe_df["Test ROC"], yerr=rfe_df["Test ROC (std)"], fmt='-o', label="Test ROC", color = "crimson")
    ax.invert_xaxis()
    ax.set_xlabel("Number of features")
    ax.set_ylabel("ROC AUC")
    ax.set_title("RFE plot")
    ax.legend()
    if output_dir is not None:
        plt.savefig(f"{output_dir}/rfe_plot.jpeg", bbox_inches='tight', dpi=1200)
        plt.close()
    else:
        plt.show()

def plot_feature_importances(models, features, output_dir=None):
    feature_importances = np.ndarray([len(features), len(models)])

    for idx, model in enumerate(models):
        try:
            feature_importances[:, idx] = model.feature_importances_
        except:
            feature_importances[:, idx] = model.estimator.feature_importances_

    mean_feature_importances = np.mean(feature_importances, axis = 1)
    std_feature_importances = np.std(feature_importances, axis = 1)

    feature_importances_df = pd.DataFrame(np.concatenate((np.array(features).reshape(-1,1), mean_feature_importances.reshape(-1,1).astype(float), std_feature_importances.reshape(-1,1).astype(float)), axis=1), columns=["Feature", "Mean feature importance", "Std feature importance"])
    feature_importances_df["Mean feature importance"] = feature_importances_df["Mean feature importance"].astype(float)
    feature_importances_df["Std feature importance"] = feature_importances_df["Std feature importance"].astype(float)

    feature_importances_df.sort_values(by="Mean feature importance", ascending=True, inplace=True)

    # Create the figure and a bar plot
    _, ax = plt.subplots(figsize=(5, 6), dpi = 1000)
    ax.grid(lw=0.5, alpha=0.5)
    ax.barh(feature_importances_df["Feature"], feature_importances_df["Mean feature importance"], color="mediumblue", height=0.5, alpha=1.0, ecolor='black', capsize=2)  
    ax.set_title("Feature importance analysis (XGBoost RF Regressor)")
    ax.set_xlabel("Mean feature importance")

    if output_dir is not None:
        plt.savefig(f"{output_dir}/feature_importances.jpeg", bbox_inches='tight', dpi=1200)
    else:
        plt.show()