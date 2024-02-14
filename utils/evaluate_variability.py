import os

import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from sklearn.metrics import *

def evaluate_classification_model(preprocessed_data_list, summary_list, class_label, plot=True, output_dir=None, verbose = True):
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

    results = {}

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
    tprs_upper_train = np.minimum(mean_tpr_train + std_tpr_train, 1)
    tprs_lower_train = np.maximum(mean_tpr_train - std_tpr_train, 0)
    # tprs_upper_train = np.minimum(mean_tpr_train + 1.96 * std_tpr_train / np.sqrt(len(roc_auc_train_list)), 1)
    # tprs_lower_train = np.maximum(mean_tpr_train - 1.96 * std_tpr_train / np.sqrt(len(roc_auc_train_list)), 0)
    # tprs_lower_train, tprs_upper_train = compute_bounds_bootstrap(interp_tpr_train, 1000)
    # tprs_upper_train = np.minimum(tprs_upper_train, 1)
    # tprs_lower_train = np.maximum(tprs_lower_train, 0)
    # Compute bound of AUC in train
    roc_auc_train_lower, roc_auc_train_upper = compute_bounds_bootstrap(np.array(roc_auc_train_list), 1000)

    # Interpolate all fpr_test_list and tpr_test_list to 100 points
    mean_fpr_test = np.linspace(0, 1, 100)
    interp_tpr_test = np.ndarray([len(summary_list), len(mean_fpr_test)])
    for idx in range(len(fpr_test_list)):
        interp_tpr_test[idx] = interp(mean_fpr_test, fpr_test_list[idx], tpr_test_list[idx])
    mean_tpr_test = np.mean(interp_tpr_test, axis=0)
    mean_tpr_test[0] = 0.0
    mean_tpr_test[-1] = 1.0
    std_tpr_test = np.std(interp_tpr_test, axis=0)
    tprs_upper_test = np.minimum(mean_tpr_test + std_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - std_tpr_test, 0)
    # tprs_upper_test = np.minimum(mean_tpr_test + 1.96 * std_tpr_test / np.sqrt(len(roc_auc_test_list)), 1)
    # tprs_lower_test = np.maximum(mean_tpr_test - 1.96 * std_tpr_test / np.sqrt(len(roc_auc_test_list)), 0)
    # tprs_lower_test, tprs_upper_test = compute_bounds_bootstrap(interp_tpr_test, 1000)
    # tprs_upper_test = np.minimum(tprs_upper_test, 1)
    # tprs_lower_test = np.maximum(tprs_lower_test, 0)
    # Compute bound of AUC in test
    roc_auc_test_lower, roc_auc_test_upper = compute_bounds_bootstrap(np.array(roc_auc_test_list), 1000)

    # Compute optimal threshold by maximizing the Youden's index
    optimal_idx_test = np.argmax(mean_tpr_test - mean_fpr_test)
    optimal_tpr_test = mean_tpr_test[optimal_idx_test]
    optimal_fpr_test = mean_fpr_test[optimal_idx_test]
    optimal_sensitivity_test = optimal_tpr_test
    optimal_specificity_test = 1 - optimal_fpr_test

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
    precisions_upper_train = np.minimum(mean_precision_train + std_precision_train, 1)
    precisions_lower_train = np.maximum(mean_precision_train - std_precision_train, 0)
    
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
    precisions_upper_test = np.minimum(mean_precision_test + std_precision_test, 1)
    precisions_lower_test = np.maximum(mean_precision_test - std_precision_test, 0)

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

    if verbose:
        print("Weighted precision (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_weighted_precision, mean_weighted_precision - 1.96 * std_weighted_precision / np.sqrt(len(weighted_precision_list)), mean_weighted_precision + 1.96 * std_weighted_precision / np.sqrt(len(weighted_precision_list))))
        print("Weighted recall (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_weighted_recall, mean_weighted_recall - 1.96 * std_weighted_recall / np.sqrt(len(weighted_recall_list)), mean_weighted_recall + 1.96 * std_weighted_recall / np.sqrt(len(weighted_recall_list))))
        print("Weighted F1-score (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_weighted_f1_score, mean_weighted_f1_score - 1.96 * std_weighted_f1_score / np.sqrt(len(weighted_f1_score_list)), mean_weighted_f1_score + 1.96 * std_weighted_f1_score / np.sqrt(len(weighted_f1_score_list))))
        print("Sensitivity (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_senstivity, mean_senstivity - 1.96 * std_senstivity / np.sqrt(len(senstivity_list)), mean_senstivity + 1.96 * std_senstivity / np.sqrt(len(senstivity_list))))
        print("Specificity (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_specificity, mean_specificity - 1.96 * std_specificity / np.sqrt(len(specificity_list)), mean_specificity + 1.96 * std_specificity / np.sqrt(len(specificity_list))))
        print("F1-score (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_f1_score, mean_f1_score - 1.96 * std_f1_score / np.sqrt(len(f1_score_list)), mean_f1_score + 1.96 * std_f1_score / np.sqrt(len(f1_score_list))))
        print("FPR (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_fpr, mean_fpr - 1.96 * std_fpr / np.sqrt(len(fpr_list)), mean_fpr + 1.96 * std_fpr / np.sqrt(len(fpr_list))))
        print("FNR (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_fnr, mean_fnr - 1.96 * std_fnr / np.sqrt(len(fnr_list)), mean_fnr + 1.96 * std_fnr / np.sqrt(len(fnr_list))))
        print("NPV (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_npv, mean_npv - 1.96 * std_npv / np.sqrt(len(npv_list)), mean_npv + 1.96 * std_npv / np.sqrt(len(npv_list))))
        print("PPV (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_ppv, mean_ppv - 1.96 * std_ppv / np.sqrt(len(ppv_list)), mean_ppv + 1.96 * std_ppv / np.sqrt(len(ppv_list))))
        print("Cohen's kappa (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_cohens_kappa, mean_cohens_kappa - 1.96 * std_cohens_kappa / np.sqrt(len(cohens_kappa_list)), mean_cohens_kappa + 1.96 * std_cohens_kappa / np.sqrt(len(cohens_kappa_list))))
        print("Matthews correlation coefficient (95%CI): {:.2f} ({:.2f}-{:.2f})".format(mean_matthews_corrcoef, mean_matthews_corrcoef - 1.96 * std_matthews_corrcoef / np.sqrt(len(matthews_corrcoef_list)), mean_matthews_corrcoef + 1.96 * std_matthews_corrcoef / np.sqrt(len(matthews_corrcoef_list))))

    # Plot average confusion matrix
    cm_mean = np.mean(np.array(cm_list), axis=0).astype(int)
    cm_std = np.std(np.array(cm_list), axis=0).astype(int)
    # Normalize to 100 values
    # cm_std = cm_std / np.sum(cm_mean) * 100
    # cm_mean = cm_mean / np.sum(cm_mean) * 100
    # cm_mean = cm_mean.astype(int)
    # cm_std = cm_std.astype(int)

    # Store all results in a dict
    results = {
        "train_mean_roc_auc": np.mean(roc_auc_train_list),
        "train_std_roc_auc": np.std(roc_auc_train_list),
        "test_mean_roc_auc": np.mean(roc_auc_test_list),
        "test_std_roc_auc": np.std(roc_auc_test_list),
        "train_mean_pr_auc": np.mean(pr_auc_train_list),
        "train_std_pr_auc": np.std(pr_auc_train_list),
        "test_mean_pr_auc": np.mean(pr_auc_test_list),
        "test_std_pr_auc": np.std(pr_auc_test_list),
        "mean_weighted_precision": mean_weighted_precision,
        "std_weighted_precision": std_weighted_precision,
        "mean_weighted_recall": mean_weighted_recall,
        "std_weighted_recall": std_weighted_recall,
        "mean_weighted_f1_score": mean_weighted_f1_score,
        "std_weighted_f1_score": std_weighted_f1_score,
        "mean_senstivity": mean_senstivity,
        "std_senstivity": std_senstivity,
        "mean_specificity": mean_specificity,
        "std_specificity": std_specificity,
        "mean_f1_score": mean_f1_score,
        "std_f1_score": std_f1_score,
        "mean_fpr": mean_fpr,
        "std_fpr": std_fpr,
        "mean_fnr": mean_fnr,
        "std_fnr": std_fnr,
        "mean_npv": mean_npv,
        "std_npv": std_npv,
        "mean_ppv": mean_ppv,
        "std_ppv": std_ppv,
        "mean_cohens_kappa": mean_cohens_kappa,
        "std_cohens_kappa": std_cohens_kappa,
        "mean_matthews_corrcoef": mean_matthews_corrcoef,
        "std_matthews_corrcoef": std_matthews_corrcoef
    }

    if plot:
        # Plot the ROC curve
        plt.figure(figsize=(5, 5), dpi = 100)
        plt.grid(lw=0.5, alpha=0.5)
        plt.rcParams.update({'font.size': 14})
        # plt.plot(mean_fpr_train, mean_tpr_train, color="b", lw=2, label=f"Train ROC (AUC = {np.mean(roc_auc_train_list):.2f} [95%CI {np.mean(roc_auc_train_list) - 1.96 * np.std(roc_auc_train_list) / np.sqrt(len(roc_auc_train_list)):.2f}-{np.mean(roc_auc_train_list) + 1.96 * np.std(roc_auc_train_list) / np.sqrt(len(roc_auc_train_list)):.2f}])")
        plt.plot(mean_fpr_train, mean_tpr_train, color="mediumblue", lw=1.25, label=f"Train ROC (AUC = {np.mean(roc_auc_train_list):.2f} (95%CI {roc_auc_train_lower:.2f}-{roc_auc_train_upper:.2f})", alpha=0.5)
        plt.fill_between(mean_fpr_train, tprs_lower_train, tprs_upper_train, color="mediumblue", alpha=0.1)
        # plt.plot(mean_fpr_test, mean_tpr_test, color="r", lw=2,  label=f"Test ROC (AUC = {np.mean(roc_auc_test_list):.2f} [95%CI {np.mean(roc_auc_test_list) - 1.96 * np.std(roc_auc_test_list) / np.sqrt(len(roc_auc_test_list)):.2f}-{np.mean(roc_auc_test_list) + 1.96 * np.std(roc_auc_test_list) / np.sqrt(len(roc_auc_test_list)):.2f}])")
        plt.plot(mean_fpr_test, mean_tpr_test, color="crimson", lw=1.25,  label=f"Test ROC (AUC = {np.mean(roc_auc_test_list):.2f} (95%CI {roc_auc_test_lower:.2f}-{roc_auc_test_upper:.2f})")
        plt.fill_between(mean_fpr_test, tprs_lower_test, tprs_upper_test, color="crimson", alpha=0.2)
        plt.plot([0, 1], [0, 1], color="k", linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(fontsize = 10)
        if output_dir is not None:
            plt.savefig(f"{output_dir}/roc_curve.jpeg", bbox_inches='tight', dpi=1200)

        results = {
            "mean_fpr": list(mean_fpr_test),
            "mean_tpr": list(mean_tpr_test),
            "std_tpr": list(std_tpr_test),
            "mean_auc": np.mean(roc_auc_test_list),
            "std_auc": np.std(roc_auc_test_list),
            "tprs_upper": list(tprs_upper_test),
            "tprs_lower": list(tprs_lower_test),
            "optimal_tpr": optimal_tpr_test,
            "optimal_fpr": optimal_fpr_test,
            "optimal_sensitivity": optimal_sensitivity_test,
            "optimal_specificity": optimal_specificity_test
        }

        # Save results to a file
        import json
        if output_dir is not None:
            with open(f"{output_dir}/roc_results.json", "w") as f:
                json.dump(results, f, indent=4)

        # Plot PR curve
        plt.figure(figsize=(5, 5), dpi = 100)
        plt.grid(lw=0.5, alpha=0.5)
        plt.rcParams.update({'font.size': 14})
        plt.plot(mean_recall_train, mean_precision_train, color="mediumblue", lw=1.25, label=f"Train PR (AUC = {np.mean(pr_auc_train_list):.2f} [95%CI {np.mean(pr_auc_train_list) - 1.96 * np.std(pr_auc_train_list) / np.sqrt(len(pr_auc_train_list)):.2f}-{np.mean(pr_auc_train_list) + 1.96 * np.std(pr_auc_train_list) / np.sqrt(len(pr_auc_train_list)):.2f}])", alpha = 0.5)
        plt.fill_between(mean_recall_train, precisions_lower_train, precisions_upper_train, color="mediumblue", alpha=0.1)
        plt.plot(mean_recall_test, mean_precision_test, color="crimson", lw=1.25, label=f"Test PR (AUC = {np.mean(pr_auc_test_list):.2f} [95%CI {np.mean(pr_auc_test_list) - 1.96 * np.std(pr_auc_test_list) / np.sqrt(len(pr_auc_test_list)):.2f}-{np.mean(pr_auc_test_list) + 1.96 * np.std(pr_auc_test_list) / np.sqrt(len(pr_auc_test_list)):.2f}])")
        plt.fill_between(mean_recall_test, precisions_lower_test, precisions_upper_test, color="r", alpha=0.2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR curve")
        plt.legend(fontsize=10)
        if output_dir is not None:
            plt.savefig(f"{output_dir}/pr_curve.png", bbox_inches='tight', dpi=1200)

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

    return results