import os
import numpy as np
import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")    

def univariate_analysis(df, class_label, features_to_ignore, output_dir = None, plot_only_significant = True, show_plot = False):
    features = []
    odds_ratios = []
    p_values = []
    upper_confidence_intervals = []
    lower_confidence_intervals = []

    # Assuming df, info_columns, class_label are defined
    for feature in df.columns:
        if feature in features_to_ignore or features == class_label:
            continue
        else:
            try:    
                X = sm.add_constant(df[feature])  # adding a constant
                y = df[class_label]

                model = sm.Logit(y, X)
                result = model.fit(disp=0)  # disp=0 turns off the convergence messages

                # Odds Ratio and Confidence Interval
                params = result.params
                conf = result.conf_int()
                conf['OR'] = params
                conf.columns = ['2.5%', '97.5%', 'OR']
                conf = np.exp(conf)

                # P-value
                p_value = result.pvalues[feature]

                # print(f"Odds ratio for {feature} is {conf['OR'][feature]:.3f} with 95% CI ({conf['2.5%'][feature]:.3f}, {conf['97.5%'][feature]:.3f}) and p-value {p_value:.3f}")

                odds_ratios.append(conf['OR'][feature])
                p_values.append(p_value)
                upper_confidence_intervals.append(conf['97.5%'][feature])
                lower_confidence_intervals.append(conf['2.5%'][feature])
                features.append(feature)
            except Exception as e:
                print(f"Error in {feature}: {e}")
                continue


    odds_ratios = np.array(odds_ratios)
    p_values = np.array(p_values)
    upper_confidence_intervals = np.array(upper_confidence_intervals)
    lower_confidence_intervals = np.array(lower_confidence_intervals)

    # Sort features by odds ratio
    odds_ratios_sorted = odds_ratios[np.argsort(odds_ratios)]
    p_values_sorted = p_values[np.argsort(odds_ratios)]
    upper_confidence_intervals_sorted = upper_confidence_intervals[np.argsort(odds_ratios)]
    lower_confidence_intervals_sorted = lower_confidence_intervals[np.argsort(odds_ratios)]
    features_sorted = np.array(features)[np.argsort(odds_ratios)]

    # Create a dataframe with the results
    or_df = pd.DataFrame({
                        "Feature": features_sorted,
                        "Odds ratio": odds_ratios_sorted,
                        "Lower confidence interval": lower_confidence_intervals_sorted,
                        "Upper confidence interval": upper_confidence_intervals_sorted,
                        "p-value": p_values_sorted
                        })

    if plot_only_significant:
        # Plot features is p value < 0.1
        odds_ratios_sorted_ = odds_ratios_sorted[p_values_sorted < 0.1]
        upper_confidence_intervals_sorted_ = upper_confidence_intervals_sorted[p_values_sorted < 0.1]
        lower_confidence_intervals_sorted_ = lower_confidence_intervals_sorted[p_values_sorted < 0.1]
        features_sorted_ = features_sorted[p_values_sorted < 0.1]
        plt.figure(figsize=(10, max(3, len(odds_ratios_sorted_) / 2)))
        plt.errorbar(odds_ratios_sorted_, np.arange(len(odds_ratios_sorted_)), xerr=[odds_ratios_sorted_ - lower_confidence_intervals_sorted_, upper_confidence_intervals_sorted_ - odds_ratios_sorted_], fmt='o')
        plt.axvline(1, color='k', linestyle='--')
        plt.xlabel("Odds ratio")
        plt.ylabel("Feature")
        plt.yticks(np.arange(len(odds_ratios_sorted_)), features_sorted_)
        plt.title(f"Odds ratios for {class_label}")
        # Print the p-value on the right side, outside the plot
        for i in range(len(p_values_sorted[p_values_sorted < 0.1])):
            plt.text(np.max(upper_confidence_intervals_sorted[p_values_sorted < 0.1]) + 0.07 * np.max(upper_confidence_intervals_sorted), i, f"p={p_values_sorted[p_values_sorted < 0.1][i]:.3f}", verticalalignment='center')
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 10))
        plt.errorbar(odds_ratios_sorted, np.arange(len(odds_ratios_sorted)), xerr=[odds_ratios_sorted - lower_confidence_intervals_sorted, upper_confidence_intervals_sorted - odds_ratios_sorted], fmt='o')
        plt.axvline(1, color='k', linestyle='--')
        plt.xlabel("Odds ratio")
        plt.ylabel("Feature")
        plt.yticks(np.arange(len(odds_ratios_sorted)), features_sorted)
        plt.title(f"Odds ratios for {class_label}")
        # Print the p-value on the right side, outside the plot
        for i in range(len(p_values_sorted)):
            plt.text(np.max(upper_confidence_intervals_sorted) + 0.07 * np.max(upper_confidence_intervals_sorted), i, f"p={p_values_sorted[i]:.3f}", verticalalignment='center')
        plt.tight_layout()
        
    if output_dir is not None:
        os.makedirs(output_dir + f"/univariate_analysis", exist_ok=True)
        plt.savefig(output_dir + f"/univariate_analysis/univariate_analysis_{class_label}.jpeg")
        # Save or_df
        or_df.to_csv(output_dir + f"/univariate_analysis/univariate_analysis_{class_label}.csv", index=False)

    if show_plot:
        plt.show()
    else:
        plt.close()

    significant_features = features_sorted[p_values_sorted < 0.1]

    return or_df, significant_features