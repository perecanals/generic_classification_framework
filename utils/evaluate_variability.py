import os

import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.metrics import *

def draw_roc_train_validation(preprocessed_data_list, summary_list, class_label, output_dir=None):
    fpr_train_list = []
    tpr_train_list = []
    fpr_test_list = []
    tpr_test_list = []
    precision_train_list = []
    precision_test_list = []
    recall_train_list = []
    recall_test_list = []

    roc_auc_train_list = []
    roc_auc_test_list = []
    pr_auc_train_list = []
    pr_auc_test_list = []

    cm_list = []

    weighted_precision_list = []
    weighted_recall_list = []
    weighted_f1_score_list = []
    senstivity_list = []
    specificity_list = []
    f1_score_list = []
    fpr_list = []
    fnr_list = []
    npv_list = []
    ppv_list = []
    cohens_kappa_list = []
    matthews_corrcoef_list = []

    for idx, summary in enumerate(summary_list):

        preprocessed_data = preprocessed_data_list[idx]

        y_train = preprocessed_data["y_train"]
        y_test = preprocessed_data["y_test"]
        y_test_pred = summary["y_test_pred"]

        fpr_train_list.append(summary["fpr_train"])
        tpr_train_list.append(summary["tpr_train"])
        fpr_test_list.append(summary["fpr_test"])
        tpr_test_list.append(summary["tpr_test"])

        precision_train, recall_train, _ = precision_recall_curve(y_train, summary["probs_train"])
        precision_train_list.append(precision_train)
        recall_train_list.append(recall_train)
        precision_test, recall_test, _ = precision_recall_curve(y_test, summary["probs_test"])
        precision_test_list.append(precision_test)
        recall_test_list.append(recall_test)

        # Compute PR AUC with precision and recall
        pr_auc_train = auc(recall_train, precision_train)
        pr_auc_test = auc(recall_test, precision_test)

        roc_auc_train_list.append(summary["roc_auc_train"])
        roc_auc_test_list.append(summary["roc_auc_test"])
        pr_auc_train_list.append(pr_auc_train)
        pr_auc_test_list.append(pr_auc_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        cm_list.append(cm)
        tn, fp, fn, tp = cm.ravel()

        weighted_precision_list.append(precision_score(y_test, y_test_pred, average = "weighted"))
        weighted_recall_list.append(recall_score(y_test, y_test_pred, average = "weighted"))
        weighted_f1_score_list.append(f1_score(y_test, y_test_pred, average = "weighted"))
        senstivity_list.append(tp / (tp + fn))
        specificity_list.append(tn / (tn + fp))
        f1_score_list.append(2 * tp / (2 * tp + fp + fn))
        fpr_list.append(fp / (fp + tn))
        fnr_list.append(fn / (fn + tp))
        npv_list.append(tn / (tn + fn))
        ppv_list.append(tp / (tp + fp))
        cohens_kappa_list.append(cohen_kappa_score(y_test, y_test_pred))
        matthews_corrcoef_list.append(matthews_corrcoef(y_test, y_test_pred))

    def compute_bounds_bootstrap(values, n_bootstrap_samples):
        lower_bounds = []
        upper_bounds = []
        if len(values.shape) == 1:
            # extend to 2D
            values = np.expand_dims(values, axis=1)
        for idx in range(values.shape[1]):
            data = values[:, idx]
            # Perform bootstrapping
            bootstrap_statistics = []
            for _ in range(n_bootstrap_samples):
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                statistic = np.mean(bootstrap_sample)
                bootstrap_statistics.append(statistic)

            # Compute the 95% CI
            lower_bounds.append(np.percentile(bootstrap_statistics, 2.5))
            upper_bounds.append(np.percentile(bootstrap_statistics, 97.5))

        if len(lower_bounds) == 1:
            return lower_bounds[0], upper_bounds[0]
        else:
            return np.array(lower_bounds), np.array(upper_bounds)

    # Interpolate all fpr_train_list and tpr_train_list to 100 points
    mean_fpr_train = np.linspace(0, 1, 100)
    interp_tpr_train = np.ndarray([len(summary_list), len(mean_fpr_train)])
    for idx in range(len(fpr_train_list)):
        interp_tpr_train[idx] = interp(mean_fpr_train, fpr_train_list[idx], tpr_train_list[idx])
    mean_tpr_train = np.mean(interp_tpr_train, axis=0)
    mean_tpr_train[0] = 0.0
    mean_tpr_train[-1] = 1.0
    std_tpr_train = np.std(interp_tpr_train, axis=0)
    tprs_upper_train = np.minimum(mean_tpr_train + 1.96 * std_tpr_train, 1)
    tprs_lower_train = np.maximum(mean_tpr_train - 1.96 * std_tpr_train, 0)
    roc_auc_train_lower = np.mean(roc_auc_train_list) - 1.96 * np.std(roc_auc_train_list)
    roc_auc_train_upper = np.mean(roc_auc_train_list) + 1.96 * np.std(roc_auc_train_list)
    # tprs_lower_train, tprs_upper_train = compute_bounds_bootstrap(interp_tpr_train, 1000)
    # tprs_upper_train = np.minimum(tprs_upper_train, 1)
    # tprs_lower_train = np.maximum(tprs_lower_train, 0)
    # # Compute bound of AUC in train
    # roc_auc_train_lower, roc_auc_train_upper = compute_bounds_bootstrap(np.array(roc_auc_train_list), 1000)

    # Interpolate all fpr_test_list and tpr_test_list to 100 points
    mean_fpr_test = np.linspace(0, 1, 100)
    interp_tpr_test = np.ndarray([len(summary_list), len(mean_fpr_test)])
    for idx in range(len(fpr_test_list)):
        interp_tpr_test[idx] = interp(mean_fpr_test, fpr_test_list[idx], tpr_test_list[idx])
    mean_tpr_test = np.mean(interp_tpr_test, axis=0)
    mean_tpr_test[0] = 0.0
    mean_tpr_test[-1] = 1.0
    std_tpr_test = np.std(interp_tpr_test, axis=0)
    tprs_upper_test = np.minimum(mean_tpr_test + 1.96 * std_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - 1.96 * std_tpr_test, 0)
    roc_auc_test_lower = np.mean(roc_auc_test_list) - 1.96 * np.std(roc_auc_test_list)
    roc_auc_test_upper = np.mean(roc_auc_test_list) + 1.96 * np.std(roc_auc_test_list)
    # tprs_lower_test, tprs_upper_test = compute_bounds_bootstrap(interp_tpr_test, 1000)
    # tprs_upper_test = np.minimum(tprs_upper_test, 1)
    # tprs_lower_test = np.maximum(tprs_lower_test, 0)
    # # Compute bound of AUC in test
    # roc_auc_test_lower, roc_auc_test_upper = compute_bounds_bootstrap(np.array(roc_auc_test_list), 1000)

    # Plot the ROC curve
    plt.figure(figsize=(5, 5), dpi = 100)
    plt.rcParams.update({'font.size': 14})
    plt.plot(mean_fpr_train, mean_tpr_train, color="b", lw=2, label=f"Train ROC (AUC = {np.mean(roc_auc_train_list):.2f} (95% CI {roc_auc_train_lower:.2f} - {roc_auc_train_upper:.2f})")
    plt.fill_between(mean_fpr_train, tprs_lower_train, tprs_upper_train, color="b", alpha=0.2)
    plt.plot(mean_fpr_test, mean_tpr_test, color="r", lw=2,  label=f"Test ROC (AUC = {np.mean(roc_auc_test_list):.2f} (95% CI {roc_auc_test_lower:.2f} - {roc_auc_test_upper:.2f})")
    plt.fill_between(mean_fpr_test, tprs_lower_test, tprs_upper_test, color="r", alpha=0.2)
    plt.plot([0, 1], [0, 1], color="k", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve for {class_label} classification")
    plt.legend(loc="lower right")
    if output_dir is not None:
        plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches='tight', dpi=1200)

    # Interpolate all precision_train_list and recall_train_list to 100 points
    mean_recall_train = np.linspace(0, 1, 100)
    interp_precision_train = np.ndarray([len(summary_list), len(mean_recall_train)])
    for idx in range(len(precision_train_list)):
        interp_precision_train[idx] = interp(mean_recall_train, np.flip(recall_train_list[idx]), np.flip(precision_train_list[idx]))
    mean_recall_train  = np.flip(mean_recall_train)
    interp_precision_train = np.flip(interp_precision_train, axis=1)
    mean_precision_train = np.mean(interp_precision_train, axis=0)
    mean_precision_train[0] = 0.0
    mean_precision_train[-1] = 1.0
    std_precision_train = np.std(interp_precision_train, axis=0)
    precisions_upper_train = np.minimum(mean_precision_train + 1.96 * std_precision_train, 1)
    precisions_lower_train = np.maximum(mean_precision_train - 1.96 * std_precision_train, 0)
    pr_auc_train_lower = np.mean(pr_auc_train_list) - 1.96 * np.std(pr_auc_train_list)
    pr_auc_train_upper = np.mean(pr_auc_train_list) + 1.96 * np.std(pr_auc_train_list)
    # precisions_lower_train, precisions_upper_train = compute_bounds_bootstrap(interp_precision_train, 1000)
    # precisions_upper_train = np.minimum(precisions_upper_train, 1)
    # precisions_lower_train = np.maximum(precisions_lower_train, 0)
    # # Compute bound of AUC in train
    # pr_auc_train_lower, pr_auc_train_upper = compute_bounds_bootstrap(np.array(pr_auc_train_list), 1000)
    
    # Interpolate all precision_test_list and recall_test_list to 100 points
    mean_recall_test = np.linspace(0, 1, 100)
    interp_precision_test = np.ndarray([len(summary_list), len(mean_recall_test)])
    for idx in range(len(precision_test_list)):
        interp_precision_test[idx] = interp(mean_recall_test, np.flip(recall_test_list[idx]), np.flip(precision_test_list[idx]))
    mean_recall_test  = np.flip(mean_recall_test)
    interp_precision_test = np.flip(interp_precision_test, axis=1)
    mean_precision_test = np.mean(interp_precision_test, axis=0)
    mean_precision_test[0] = 0.0
    mean_precision_test[-1] = 1.0
    std_precision_test = np.std(interp_precision_test, axis=0)
    precisions_upper_test = np.minimum(mean_precision_test + 1.96 * std_precision_test, 1)
    precisions_lower_test = np.maximum(mean_precision_test - 1.96 * std_precision_test, 0)
    pr_auc_test_lower = np.mean(pr_auc_test_list) - 1.96 * np.std(pr_auc_test_list)
    pr_auc_test_upper = np.mean(pr_auc_test_list) + 1.96 * np.std(pr_auc_test_list)
    # precisions_lower_test, precisions_upper_test = compute_bounds_bootstrap(interp_precision_test, 1000)
    # precisions_upper_test = np.minimum(precisions_upper_test, 1)
    # precisions_lower_test = np.maximum(precisions_lower_test, 0)
    # # Compute bound of AUC in test
    # pr_auc_test_lower, pr_auc_test_upper = compute_bounds_bootstrap(np.array(pr_auc_test_list), 1000)

    # Plot PR curve
    plt.figure(figsize=(5, 5), dpi = 100)
    plt.rcParams.update({'font.size': 14})
    plt.plot(mean_recall_train, mean_precision_train, color="b", lw=2, label=f"Train PR (AUC = {np.mean(pr_auc_train_list):.2f} (95% CI {pr_auc_train_lower:.2f} - {pr_auc_train_upper:.2f})")
    plt.fill_between(mean_recall_train, precisions_lower_train, precisions_upper_train, color="b", alpha=0.2)
    plt.plot(mean_recall_test, mean_precision_test, color="r", lw=2,  label=f"Test PR (AUC = {np.mean(pr_auc_test_list):.2f} (95% CI {pr_auc_test_lower:.2f} - {pr_auc_test_upper:.2f})")
    plt.fill_between(mean_recall_test, precisions_lower_test, precisions_upper_test, color="r", alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve for {class_label} classification")
    plt.legend(loc="upper right")
    if output_dir is not None:
        plt.savefig(f"{output_dir}/pr_curve.png", bbox_inches='tight', dpi=1200)

    # Compute mean and std of all metrics
    mean_weighted_precision = np.mean(weighted_precision_list)
    std_weighted_precision = np.std(weighted_precision_list)
    mean_weighted_recall = np.mean(weighted_recall_list)
    std_weighted_recall = np.std(weighted_recall_list)
    mean_weighted_f1_score = np.mean(weighted_f1_score_list)
    std_weighted_f1_score = np.std(weighted_f1_score_list)
    mean_senstivity = np.mean(senstivity_list)
    std_senstivity = np.std(senstivity_list)
    mean_specificity = np.mean(specificity_list)
    std_specificity = np.std(specificity_list)
    mean_f1_score = np.mean(f1_score_list)
    std_f1_score = np.std(f1_score_list)
    mean_fpr = np.mean(fpr_list)
    std_fpr = np.std(fpr_list)
    mean_fnr = np.mean(fnr_list)
    std_fnr = np.std(fnr_list)
    mean_npv = np.mean(npv_list)
    std_npv = np.std(npv_list)
    mean_ppv = np.mean(ppv_list)
    std_ppv = np.std(ppv_list)
    mean_cohens_kappa = np.mean(cohens_kappa_list)
    std_cohens_kappa = np.std(cohens_kappa_list)
    mean_matthews_corrcoef = np.mean(matthews_corrcoef_list)
    std_matthews_corrcoef = np.std(matthews_corrcoef_list)

    print("Weighted precision: {:.2f} ± {:.2f}".format(mean_weighted_precision, std_weighted_precision))
    print("Weighted recall: {:.2f} ± {:.2f}".format(mean_weighted_recall, std_weighted_recall))
    print("Weighted F1-score: {:.2f} ± {:.2f}".format(mean_weighted_f1_score, std_weighted_f1_score))
    print("Sensitivity: {:.2f} ± {:.2f}".format(mean_senstivity, std_senstivity))
    print("Specificity: {:.2f} ± {:.2f}".format(mean_specificity, std_specificity))
    print("F1-score: {:.2f} ± {:.2f}".format(mean_f1_score, std_f1_score))
    print("FPR: {:.2f} ± {:.2f}".format(mean_fpr, std_fpr))
    print("FNR: {:.2f} ± {:.2f}".format(mean_fnr, std_fnr))
    print("NPV: {:.2f} ± {:.2f}".format(mean_npv, std_npv))
    print("PPV: {:.2f} ± {:.2f}".format(mean_ppv, std_ppv))
    print("Cohen's kappa: {:.2f} ± {:.2f}".format(mean_cohens_kappa, std_cohens_kappa))
    print("Matthews correlation coefficient: {:.2f} ± {:.2f}".format(mean_matthews_corrcoef, std_matthews_corrcoef))

    # Plot average confusion matrix
    cm_mean = np.mean(np.array(cm_list), axis=0).astype(int)
    cm_std = np.std(np.array(cm_list), axis=0).astype(int)
    # Normalize to 100 values
    # cm_std = cm_std / np.sum(cm_mean) * 100
    # cm_mean = cm_mean / np.sum(cm_mean) * 100
    # cm_mean = cm_mean.astype(int)
    # cm_std = cm_std.astype(int)

    # Plot confusion matrix
    plt.figure()
    sns.heatmap(cm_mean, cmap=plt.cm.Blues, fmt='g')
    plt.title('Confusion matrix (mean±SD) for validation')
    plt.ylabel(f'True {class_label}')
    plt.xlabel(f'Predicted {class_label}')
    # Set text at the center of the cells (for some reason it does not print these)
    plt.text(0.33, 0.5, f"{cm_mean[0, 0]} ± {cm_std[0, 0]}", fontsize=12, color="white")
    plt.text(1.38, 0.5, f"{cm_mean[0, 1]} ± {cm_std[0, 1]}", fontsize=12, color="black")
    plt.text(0.38, 1.5, f"{cm_mean[1, 0]} ± {cm_std[1, 0]}", fontsize=12, color="black")
    plt.text(1.38, 1.5, f"{cm_mean[1, 1]} ± {cm_std[1, 1]}", fontsize=12, color="black")
    if output_dir is not None:
        plt.savefig(f"{output_dir}/confusion_matrix.png", bbox_inches='tight', dpi=1200)