import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def rfe_plot(rfe_df, n_splits_rfe, output_dir=None):
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
    # Convert columns to numeric if they are not already
    rfe_df["N_features"] = pd.to_numeric(rfe_df["N_features"], errors='coerce')
    rfe_df["Train ROC"] = pd.to_numeric(rfe_df["Train ROC"], errors='coerce')
    rfe_df["Train ROC (std)"] = pd.to_numeric(rfe_df["Train ROC (std)"], errors='coerce')
    rfe_df["Test ROC"] = pd.to_numeric(rfe_df["Test ROC"], errors='coerce')
    rfe_df["Test ROC (std)"] = pd.to_numeric(rfe_df["Test ROC (std)"], errors='coerce')
    _, ax = plt.subplots(figsize=(12, 3), dpi=1000)
    ax.grid(lw=0.5, alpha=0.5)
    if len(rfe_df) == 1:
        ax.errorbar(rfe_df["N_features"], rfe_df["Train ROC"], yerr=1.96 * rfe_df["Train ROC (std)"] / np.sqrt(n_splits_rfe), fmt='-o', label="Train AUROC", color = "mediumblue", lw=1, markersize=3, alpha=0.8)
        ax.errorbar(rfe_df["N_features"], rfe_df["Test ROC"], yerr=1.96 * rfe_df["Test ROC (std)"] / np.sqrt(n_splits_rfe), fmt='-o', label="Val AUROC", color = "crimson", lw=1, markersize=3, alpha=0.8)
    else:
        # ax.errorbar(rfe_df["N_features"], rfe_df["Train ROC"], yerr=1.96 * rfe_df["Train ROC (std)"] / np.sqrt(n_splits_rfe), fmt='-o', label="Train ROC", color = "mediumblue", lw=1, markersize=3, alpha=0.8)
        # ax.errorbar(rfe_df["N_features"], rfe_df["Test ROC"], yerr=1.96 * rfe_df["Test ROC (std)"] / np.sqrt(n_splits_rfe), fmt='-o', label="Val ROC", color = "crimson", lw=1, markersize=3, alpha=0.8)
        # Plotting Train ROC
        ax.plot(rfe_df["N_features"], rfe_df["Train ROC"], '-o', label="Train AUROC", color="mediumblue", lw=1, alpha=0.8, markersize=3)
        ax.fill_between(rfe_df["N_features"], 
                        rfe_df["Train ROC"] - 1.96 * rfe_df["Train ROC (std)"] / np.sqrt(n_splits_rfe), 
                        rfe_df["Train ROC"] + 1.96 * rfe_df["Train ROC (std)"] / np.sqrt(n_splits_rfe), 
                        color="mediumblue", alpha=0.1)

        # Plotting Test/Validation ROC
        ax.plot(rfe_df["N_features"], rfe_df["Test ROC"], '-o', label="Val AUROC", color="crimson", lw=1, alpha=0.8, markersize=3)
        ax.fill_between(rfe_df["N_features"], 
                        rfe_df["Test ROC"] - 1.96 * rfe_df["Test ROC (std)"] / np.sqrt(n_splits_rfe), 
                        rfe_df["Test ROC"] + 1.96 * rfe_df["Test ROC (std)"] / np.sqrt(n_splits_rfe), 
                        color="crimson", alpha=0.1)
    ax.invert_xaxis()
    ax.set_xlabel("Number of features")
    ax.set_ylabel("AUROC")
    # ax.set_title("RFE plot")
    ax.legend(loc="lower left")
    if output_dir is not None:
        plt.savefig(f"{output_dir}/rfe_plot.jpeg", bbox_inches='tight', dpi=1200)
        plt.close()
    else:
        plt.show()

def plot_feature_importances(models, features, output_dir=None):
    feature_importances = np.ndarray([len(features), len(models), len(models[0])])

    for idx, model in enumerate(models):
        for idx_, model_ in enumerate(model):
            try:
                feature_importances[:, idx, idx_] = model_.feature_importances_
            except:
                feature_importances[:, idx, idx_] = model_.estimator.feature_importances_

    # Compute the mean across the last axis (ensemble axis)
    feature_importances = np.mean(feature_importances, axis=-1)

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