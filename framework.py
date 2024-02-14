import os
import pickle
import itertools

import pandas as pd
import numpy as np

from generic_classification_framework.utils.train_validation_splitting import split_data
from generic_classification_framework.utils.preprocessing import preprocess_data
from generic_classification_framework.utils.univariate_analysis import univariate_analysis
from generic_classification_framework.utils.classification_utils import build_classifier, evaluate_model
from generic_classification_framework.utils.evaluate_variability import evaluate_classification_model 
from generic_classification_framework.utils.feature_importance import rfe_plot, plot_feature_importances

class GenericClassificationFramework():
    """
    Generic classification object for a binary classification problems.
    It incorporates methods for:
    - Data splitting
    - Data preprocessing
    - Univariate analysis
    - Classifier building
    - Classifier fitting
    - Classifier evaluation
    - Variability assessment
    - Outlier assessment
    - Clinical assessment
    - Fine tuning

    SETUP:
        - Add class labels to the class_labels list (if multiple classes are 
        present are should not be taken into account).
        - Add feature groups as list if some kind of grouped feature selection 
        is to be performed.
        
    Monte Carlo cross-validation is used by defult. Outlier assessment averages 
    the test predictions from all classifiers and calculates the ratio of correct
    predictions for each sample. Clinical assessment merges the outlier dataframe with some 
    dataframe containing clinical data and allows us to compute the effect of the classification
    on the clinical data.
    """
    def __init__(self, 
                 df: pd.DataFrame = None, 
                 output_dir: str = None, 
                 id_label: str = None, 
                 class_label: str = None, 
                 random_seed_split: int = 42
                 ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the data. It should have one column for
            identifiers and one or more columns for the class labels. Only class
            label will be used for binary classification target.
        output_dir : str
            Directory to store the results.
        id_label : str
            Name of the column containing the identifiers.
        class_label : str
            Name of the column containing the class labels.
        random_seed_split : int
            Random seed for the train/test split.
            
        """
        
        self.original_df = df
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok = True)
            self.problem_name = output_dir.split("/")[-1]
        else:
            self.problem_name = None
        self.id_label = id_label
        self.class_label = class_label

        # Feature groups (to be adapted for a specific task)
        self.class_labels = []
        self.info_columns = [self.id_label] + self.class_labels
        # Add list of features groups if needed
        # Ex:
        # self.cv_features = ["AGE", "SEX", ...]

        # Train/test split parameters
        self.random_seed_split = random_seed_split
        self.test_size = None
        self.train_df_list, self.test_df_list = [], []
        self.n_splits = None
        
        # Preprocessing
        self.oversampling = None
        self.normalize_numerical = False
        self.preprocessed_data_list = []

        # Univariate analysis
        self.or_df = None
        self.significant_features_from_univariate = []

        # Classifier
        self.model = None
        self.params = None
        self.model_name = None
        self.random_seed_initialization = None
        self.classifiers = []
        self.features_to_include = []
        self.is_fitted = []

        # Evaluation
        self.results = {}
        self.results_cv = None

        # Variability assessment
        self.variability_df = None
        self.variability_results = None

        # Feature selection
        self.selected_features = None

        # Outlier assessment
        self.outlier_df = None

        # Clinical assessment
        self.clinical_df = None
        self.columns_to_assess = None
        self.clinical_results_df = None

        # Fine tuning
        self.fine_tuning_df = None
        self.best_params = None

    def load_data(self, df: pd.DataFrame):
        self.original_df = df

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok = True)
        os.makedirs(output_dir, exist_ok = True)
        self.model_name = output_dir.split("/")[-1]

    def set_id_label(self, id_label: str):
        self.id_label = id_label

    def set_class_label(self, class_label: str):
        self.class_label = class_label

    def get_model_name(self):

        model_name = self.model + "-"
        for key, value in self.params.items():
            # If value is float, round to 2 decimal
            if isinstance(value, float):
                model_name += key + "_" + f"{value:.2f}" + "-"
            else:
                model_name += key + "_" + str(value) + "-"
        if self.oversampling is not None:
            model_name += "oversampling_" + self.oversampling + "-"

        self.model_name = model_name[:-1]

        return self.model_name

    def split_data_train_test(self, 
                              test_size: float = 0.3, 
                              n_splits: int = 10, 
                              verbose: bool = True):
        """
        Split the data into train and test sets using stratified sampling.
        Designed for Monte Carlo cross-validation.

        Stores the train and test sets in the train_df_list and test_df_list and in
        the output_dir if it is not None.

        Parameters
        ----------
        test_size : float
            Proportion of the test set.
        n_splits : int
            Number of splits to perform.
        verbose : bool
            Whether to print information about the splits.
        """

        assert self.original_df is not None, "Please, load a dataframe using the load_data method"
        assert self.class_label is not None, "Please, set a class_label using the set_class_label method"

        self.test_size = test_size
        self.n_splits = n_splits

        for idx in range(n_splits):
            if os.path.exists((self.output_dir + f"/folds/{idx}/train_df_ratio{self.test_size}.csv")):
                print(f"Loading data for split {idx} from disk...")
                train_df = pd.read_csv(self.output_dir + f"/folds/{idx}/train_df_ratio{test_size}.csv")
                test_df = pd.read_csv(self.output_dir + f"/folds/{idx}/test_df_ratio{test_size}.csv")
            else:
                print(f"\nSplitting data for split {idx}")
                train_df, test_df = split_data(self.original_df, 
                                               self.class_label, 
                                               self.test_size, 
                                               self.random_seed_split + idx, 
                                               verbose = verbose)
                if self.output_dir is not None:
                    os.makedirs(self.output_dir + f"/folds/{idx}", exist_ok = True)
                    train_df.to_csv(self.output_dir + f"/folds/{idx}/train_df_ratio{test_size}.csv", index = False)
                    test_df.to_csv(self.output_dir + f"/folds/{idx}/test_df_ratio{test_size}.csv", index = False)

            self.train_df_list.append(train_df)
            self.test_df_list.append(test_df)

    def preprocess_data(self, 
                        oversampling: str = None, 
                        normalize_numerical: bool = False):
        """
        Preprocess the data for each split. It allows us to define the oversampling technique and
        whether to normalize the numerical features.

        Stores the preprocessed data in the preprocessed_data_list and in the output_dir if it is not None.

        To be used after the split_data_train_test method.

        Parameters
        ----------
        oversampling : str
            Name of the oversampling technique to use. If None, no oversampling is performed.
        normalize_numerical : bool
            Whether to normalize the numerical features.

        """

        assert self.train_df_list is not None, "Please, first split the data using the split_data_train_test method"

        self.oversampling = oversampling
        self.normalize_numerical = normalize_numerical

        for idx in range(self.n_splits):
            if os.path.exists((self.output_dir + f"/folds/{idx}/train_df_ratio{self.test_size}.pkl")):
                print(f"Loading preprocessed data for split {idx} from disk...")
                with open(self.output_dir + f"/folds/{idx}/train_df_ratio{self.test_size}.pkl", "rb") as f:
                    preprocessed_data = pickle.load(f)
            else:
                preprocessed_data = preprocess_data(self.train_df_list[idx], 
                                                    self.test_df_list[idx], 
                                                    self.class_label,
                                                    self.info_columns,
                                                    self.oversampling, 
                                                    self.normalize_numerical)
                if self.output_dir is not None:
                    with open(self.output_dir + f"/folds/{idx}/preprocessed_data_ratio{self.test_size}_oversampling{self.oversampling}.pkl", "wb") as f:
                        pickle.dump(preprocessed_data, f)

            self.preprocessed_data_list.append(preprocessed_data)

    def perform_univariate_analysis(self, 
                                    plot_only_significant: bool = True, 
                                    show_plot: bool = False):
        
        """
        Performs univariate logistic regression analysis for each feature in the dataset. It creates 
        a dataframe with the odds ratio, confidence intervals and p-values for each feature. It also
        stores the significant features in the significant_features_from_univariate attribute, which 
        contains all features with a p-value < 0.1.

        It can be used as a first step for feature selection.

        Parameters
        ----------
        plot_only_significant : bool    
            Whether to plot only the significant features.
        show_plot : bool
            Whether to show the plot.

        """

        assert self.original_df is not None, "Please, load a dataframe using the load_data method"
        assert self.class_label is not None, "Please, set a class_label using the set_class_label method"

        if os.path.exists(self.output_dir + f"/univariate_analysis/univariate_analysis_{self.class_label}.xlsx"):
            print("Loading univariate analysis results from disk...")
            self.or_df = pd.read_csv(self.output_dir + f"/univariate_analysis/univariate_analysis_{self.class_label}.xlsx")
            self.significant_features_from_univariate = self.or_df[self.or_df["p-value"] < 0.1]["Feature"].tolist()
        else:
            print("Performing univariate analysis...")
            self.or_df, self.significant_features_from_univariate = univariate_analysis(self.original_df,
                                                                                        self.class_label,
                                                                                        self.info_columns,
                                                                                        self.output_dir,
                                                                                        plot_only_significant,
                                                                                        show_plot)

    def build_classifier(self, 
                         model: str = "XGBRFClassifier", 
                         params: dict = None,
                         random_seed_initialization: int = 1042,
                         verbose: bool = False):
        """
        Builds basic classifier with provided parameters. Classifier should be
        one of:
        - XGBRFClassifier
        - XGBClassifier
        - RandomForestClassifier
        - RidgeClassifier
        - LogisticRegression
        - SVC

        Stores the classifier in the classifiers list and the parameters in the params attribute. It
        also creates a model name based on the parameters and stores it in the model_name attribute.

        Parameters
        ----------
        model : str
            Name of the classifier to use.
        params : dict
            Parameters to use for the classifier.
        random_seed_initialization : int
            Random seed for the classifier initialization.
        verbose : bool
            Whether to print information about the classifier.

        """
        self.model = model
        self.params = params
        self.random_seed_initialization = random_seed_initialization

        for split in range(self.n_splits):
            classifier, params = build_classifier(self.model,
                                                  self.params,
                                                  self.random_seed_initialization,
                                                  verbose = verbose)
            if len(self.classifiers) >= (split + 1):
                self.classifiers[split] = classifier
                self.is_fitted[split] = False
            else:
                self.classifiers.append(classifier)
                self.is_fitted.append(False)

            self.params = params
            self.get_model_name()
            os.makedirs(self.output_dir + f"/folds/{split}/{self.model_name}", exist_ok = True)

    def fit_classififer(self,
                        features_to_include: list = None,
                        split: int = 0,
                        # exclude_cv_features: bool = False,# Add if needed
                        verbose: bool = False):
        """
        Fits the classifier to the training data from the specified split. It uses the features_to_include
        attribute to select the features to use for the classifier. If features_to_include is None, it uses
        all features except the info_columns.

        To be used after the build_classifier method.

        Parameters
        ----------
        features_to_include : list
            List of features to include in the classifier.
        split : int
            Split to fit the classifier to.
        verbose : bool
            Whether to print information about the fitting.
        """
        def clean_feature_names(feature_names):
            for idx, feature_name in enumerate(feature_names):
                for symbol in ["<", ">", "[", "]"]:
                    if symbol in feature_name:
                        feature_names[idx] = feature_name.replace(symbol, "")
            return feature_names
        
        if features_to_include is None:
            features_to_include = [feature for feature in self.original_df.columns if feature not in self.info_columns]

        # if exclude_cv_features:
        #     features_to_include = [feature for feature in features_to_include if feature not in self.cv_features]

        self.features_to_include = features_to_include

        if verbose:
            print("Number of included features:", len(self.features_to_include))
            if len(self.features_to_include) < 10:
                print("Included features:", self.features_to_include)
            else:
                print("Included features:", self.features_to_include[:10], "...")
            print("Oversampling:", self.oversampling)
            print(f"Fitting classifier for split {split}...")

        self.features_to_include = clean_feature_names(self.features_to_include)

        X_train = self.preprocessed_data_list[split]["X_train"][self.features_to_include]
        y_train = self.preprocessed_data_list[split]["y_train"]

        self.classifiers[split].fit(X_train, y_train)
        self.is_fitted[split] = True

    def fit_classifiers_cv(self,
                           features_to_include: list = None,
                        #    exclude_cv_features: bool = False,
                           force = True,
                           verbose: bool = True):
        """
        Wrapper for the fit_classifier method. Fits the classifier to the training data from all splits
        of the cross-validation.

        Parameters
        ----------
        features_to_include : list
            List of features to include in the classifier.
        verbose : bool
            Whether to print information about the fitting.

        """
        for split in range(self.n_splits):
            if os.path.join(self.output_dir, "folds", str(split), self.get_model_name(), "results.pkl") and not force:
                print("Loading results from: ", self.output_dir, "folds", str(split), self.get_model_name(), "results.pkl")
            else:
                self.fit_classififer(features_to_include = features_to_include,
                                    split = split,
                                    #  exclude_cv_features = exclude_cv_features,
                                    verbose = verbose)
            if verbose:
                print(f"Classifier fitted for split {split}")

    def evaluate_classifier(self, 
                            split: int = 0,
                            make_plots: bool = True,
                            calibrate: bool = False,
                            verbose: bool = True):
        """
        Evaluates the classifier on the train and test data from the specified split. 
        It stores the results in the results attribute and in the output_dir if it is not None.

        To be used after the fit_classifier method.

        Parameters
        ----------
        split : int
            Split to evaluate the classifier on.
        make_plots : bool
            Whether to make plots of the evaluation.
        calibrate : bool
            Whether to calibrate the classifier.
        verbose : bool
            Whether to print information about the evaluation.

        """
        
        assert self.is_fitted[split], "Interrupting evaluation, classifier is not fitted"

        self.results[split] = {"model_name": self.model_name + f"_{split}"}

        X_train = self.preprocessed_data_list[split]["X_train"][self.features_to_include]
        y_train = self.preprocessed_data_list[split]["y_train"]
        X_test = self.preprocessed_data_list[split]["X_test"][self.features_to_include]
        y_test = self.preprocessed_data_list[split]["y_test"]

        self.results[split]["X_train"] = X_train
        self.results[split]["y_train"] = y_train
        self.results[split]["X_test"] = X_test
        self.results[split]["y_test"] = y_test

        if self.model == "RidgeClassifier":
            # # For RidgeClassifier, we have to apply the softmax function to the decision function to get probabilities
            # d_train = self.classifiers[split].decision_function(X_train)
            # d_test = self.classifiers[split].decision_function(X_test)
            # self.results[split]['probs_train'] = np.exp(d_train) / np.sum(np.exp(d_train))
            # self.results[split]['probs_test'] = np.exp(d_test) / np.sum(np.exp(d_test))
            ## Note, it did not seem to be doing something sensible with the output probabilities, so we just omit ridge classifier for now (at least the calibration)
            raise NotImplementedError
        else:
            if calibrate:
                from sklearn.calibration import CalibratedClassifierCV
                calibrated_clf = CalibratedClassifierCV(self.classifiers[split], method='sigmoid', cv='prefit')
                calibrated_clf.fit(X_test, y_test)
                self.classifiers[split] = calibrated_clf
            self.results[split]['probs_train'] = self.classifiers[split].predict_proba(X_train)[:, 1]
            self.results[split]['probs_test'] = self.classifiers[split].predict_proba(X_test)[:, 1]

        self.results[split] = evaluate_model(self.model_name,
                                             self.results[split],
                                             make_plots,
                                             self.output_dir + f"/folds/{split}/{self.model_name}",
                                             verbose = verbose)
        
        with open(self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl", "wb") as f:
            pickle.dump(self.results[split], f)

    def evaluate_classifiers_cv(self,
                                make_plots: bool = True,
                                force_evaluation: bool = False,
                                calibrate: bool = False,
                                verbose: bool = True):
        """
        Wrapper for the evaluate_classifier method. Evaluates the classifier on the train and test data from all splits
        of the cross-validation.

        Parameters
        ----------
        make_plots : bool
            Whether to make plots of the evaluation.
        force_evaluation : bool
            Whether to force the evaluation even if the results are already stored.
        calibrate : bool
            Whether to calibrate the classifier.
        verbose : bool
            Whether to print information about the evaluation.

        """
        
        for split in range(self.n_splits):
            if os.path.exists(self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl") and not force_evaluation:
                if verbose:
                    print("Loading results from: ", self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl")
                with open(self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl", "rb") as f:
                    self.results[split] = pickle.load(f)
            else:
                self.evaluate_classifier(split = split,
                                        make_plots = make_plots,
                                        calibrate = calibrate,
                                        verbose = verbose)
                
        self.results_cv = evaluate_classification_model(self.preprocessed_data_list, list(self.results.values()), self.class_label, plot=False, verbose=False)

    def assess_variability_cv(self, 
                              force: bool =False):
        """
        Method to assess variability of the classification results across the different splits of the cross-validation.

        It stores the results in the variability_df and variability_results attributes and in the output_dir if it is not None.

        To be used after the evaluate_classifiers_cv method.

        """
        self.variability_df = pd.DataFrame(columns=["Random seed (splitting)", "Train ROC", "Test ROC", "Train F1-score", "Test F1-score", "Test F1-score (train)"])

        for split in range(self.n_splits):
            if (split not in self.results and os.path.exists(self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl")) or force:
                print("Loading results from: ", self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl")
                with open(self.output_dir + f"/folds/{split}/{self.model_name}/results.pkl", "rb") as f:
                    self.results[split] = pickle.load(f)
            
            variability_line = pd.DataFrame({"Random seed (splitting)": split,
                                             "Random seed (initialization)": self.random_seed_initialization,
                                             "Train ROC":  self.results[split]["roc_auc_train"],
                                             "Test ROC":  self.results[split]["roc_auc_test"],
                                             "Train F1-score":  self.results[split]["f1_score_train"],
                                             "Test F1-score":  self.results[split]["f1_score_test_train"],
                                             "Test F1-score (train)":  self.results[split]["f1_score_test"]}, index = [0])
            
            self.variability_df = pd.concat([self.variability_df, variability_line], ignore_index=True)

        self.variability_results = self.variability_df.groupby("Random seed (initialization)").agg({
                                                               "Train ROC": ["mean", "std"],
                                                               "Test ROC": ["mean", "std"],
                                                               "Train F1-score": ["mean", "std"],
                                                               "Test F1-score": ["mean", "std"],
                                                               "Test F1-score (train)": ["mean", "std"]})

    def display_results(self):
        """
        Method to display the variability results in a table and plot, as well as print several classification metrics.
        """
        # if self.variability_results is not None:
        #     display(self.variability_results)

        os.makedirs(self.output_dir + f"/aggregated_test_size{self.test_size}/{self.model_name}", exist_ok = True)
        evaluate_classification_model(preprocessed_data_list=self.preprocessed_data_list, 
                                      summary_list=list(self.results.values()), 
                                      class_label=self.class_label, 
                                      plot=True,
                                      output_dir=self.output_dir + f"/aggregated_test_size{self.test_size}/{self.model_name}")

    def perform_rfe_cv(self, 
                       n_features_to_select: int = 100, 
                       step = 0.3, 
                       n_splits_rfe: int = 5, 
                       verbose: bool = False):
        """
        Perform recursive feature elimination (RFE) across multiple data splits and eliminate features
        that are consistently ranked as least important in all splits.

        Parameters
        ----------
        n_features_to_select : int
            Number of features to select.
        step : float
            Fraction of features to eliminate at each step.
        n_splits_rfe : int
            Number of splits to perform RFE on.
        verbose : bool
            Whether to print information about the RFE.

        """

        from sklearn.feature_selection import RFE
        from collections import defaultdict

        assert len(self.train_df_list) >= n_splits_rfe, "Not enough data splits available."

        # Dictionary to keep track of feature rankings across splits
        feature_rankings = defaultdict(list)

        # Perform RFE for each split and store the rankings
        for split in range(n_splits_rfe):
            X_train = self.preprocessed_data_list[split]["X_train"][self.features_to_include]
            y_train = self.preprocessed_data_list[split]["y_train"]

            # Initialize RFE with the given estimator and parameters
            estimator, _ = build_classifier(self.model, self.params, self.random_seed_initialization)
            rfe = RFE(estimator=estimator, 
                      n_features_to_select=n_features_to_select, 
                      step=step, 
                      verbose=verbose)
            rfe.fit(X_train, y_train)

            # Store the ranking for each feature
            for feature, ranking in zip(X_train.columns, rfe.ranking_):
                feature_rankings[feature].append(ranking)

            if verbose:
                print(f"RFE completed for split {split}.")

        # Calculate average ranking for each feature
        average_rankings = {feature: sum(rankings) / len(rankings) for feature, rankings in feature_rankings.items()}

        # Select the top n_features_to_select based on average ranking
        selected_features = sorted(average_rankings, key=average_rankings.get)[:n_features_to_select]

        if verbose:
            print(f"Selected features based on average ranking: {selected_features}")

        self.selected_features = selected_features
        self.feature_rankings = average_rankings

    def perform_rfe_cv_gain(self, 
                            n_features_to_select: int = 10, 
                            step = 0.1, 
                            n_arms = 2,
                            n_splits_rfe: int = 10, 
                            initial_features: list = None,
                            make_performance_plot: bool = False,
                            n_features_fine_analysis: int = 1,
                            select_best=True):
        """
        Perform recursive feature elimination (RFE) across multiple data splits and eliminate features
        that are consistently ranked as least important in all splits.

        Parameters
        ----------
        n_features_to_select : int
            Number of features to select.
        step : float
            Fraction of features to eliminate at each step.
        n_arms : int
            Number of arms to use for the rfe algorithm.
        n_splits_rfe : int
            Number of splits to perform RFE on.
        initial_features : list
            List of initial features to start the RFE with.
        make_performance_plot : bool
            Whether to make a performance plot with the selected features.
        n_features_fine_analysis : int
            Number of features to perform a fine analysis on. If set to 1 (or less), 
            no fine analysis will be made.
        select_best : bool
            Whether to select the best features based on the fine analysis.

        """
        assert len(self.train_df_list) >= (n_arms * n_splits_rfe), "Not enough data splits available."

        if initial_features is None:
            initial_features = [feature for feature in self.preprocessed_data_list[0]["X_train"].columns]

        selected_features = initial_features

        if isinstance(n_features_to_select, float):
            n_features_to_select *= len(initial_features)
            n_features_to_select = int(n_features_to_select)

        iteration = 0

        feature_ranking_dict = {}

        original_n_splits = self.n_splits

        if make_performance_plot:
            rfe_df = pd.DataFrame(columns = ["Iteration", "N_features", "Train ROC", "Train ROC (std)", "Test ROC", "Test ROC (std)"])
            self.n_splits = min(10, self.n_splits)
            # Evaluate performance with all features
            self.build_classifier(model = self.model, params = self.params, random_seed_initialization = self.random_seed_initialization, verbose = False)
            self.fit_classifiers_cv(features_to_include=selected_features, verbose = False)
            self.evaluate_classifiers_cv(make_plots = False, force_evaluation = True, verbose = False)
            rfe_line = pd.DataFrame({"Iteration": [iteration], 
                                    "N_features": [len(selected_features)], 
                                    "Train ROC": [self.results_cv["train_mean_roc_auc"]], 
                                    "Train ROC (std)": [self.results_cv["train_std_roc_auc"]],
                                    "Test ROC": [self.results_cv["test_mean_roc_auc"]],
                                    "Test ROC (std)": [self.results_cv["test_std_roc_auc"]],
                                    "Features": [selected_features]}, index = [0])
            rfe_df = pd.concat([rfe_df, rfe_line], ignore_index = True)
            

        while len(selected_features) > n_features_to_select:
            print(f"Iteration {iteration}: {len(selected_features)} features remaining")
            feature_ranking_dict[iteration] = {}
            for arm in range(n_arms):
                print("Arm", arm, "of", n_arms)
                feature_ranking_dict[iteration][arm] = {}
                feature_ranking_dict[iteration][arm]["feature_importances"] = np.ndarray([n_splits_rfe, len(selected_features)])
                feature_ranking_dict[iteration][arm]["feature_names"] = np.array(selected_features)
                for split in range(n_splits_rfe):
                    print("\tSplit:", split)
                    X_train = self.preprocessed_data_list[arm * n_splits_rfe + split]["X_train"]
                    y_train = self.preprocessed_data_list[arm * n_splits_rfe + split]["y_train"]
                    # Initialize RFE with the given estimator and parameters
                    estimator, _ = build_classifier(self.model, self.params, self.random_seed_initialization)
                    # Fit classifier with current feature list
                    estimator.fit(X_train[selected_features], y_train)
                    # Store feature importances
                    feature_ranking_dict[iteration][arm]["feature_importances"][split, :] = estimator.feature_importances_
                # Compute mean feature importances
                feature_ranking_dict[iteration][arm]["mean_feature_importances"] = np.mean(feature_ranking_dict[iteration][arm]["feature_importances"], axis = 0)
                # Sort features by mean feature importances
                feature_ranking_dict[iteration][arm]["sorted_feature_names"] = feature_ranking_dict[iteration][arm]["feature_names"][np.argsort(feature_ranking_dict[iteration][arm]["mean_feature_importances"])]

            # Define step as either the number of features to eliminate or a fraction of the remaining features
            if isinstance(step, float):
                step = int(step * len(selected_features))

            if len(selected_features) <= n_features_fine_analysis:
                # Set a step of 1
                step = 1
            else:
                # Set a minimum step of 2
                step = max(step, 2) 

            # Set a maximum step of the remaining features (depending on the difference between the remaining 
            # features and the number of features to select)
            if len(selected_features) - step < n_features_to_select:
                step = len(selected_features) - n_features_to_select

            # Get coincidences between last step features
            coincidences = []
            for arm in range(n_arms):
                coincidences.append(feature_ranking_dict[iteration][arm]["sorted_feature_names"][:step])
            coincidences = np.concatenate(coincidences)
            coincidences, counts = np.unique(coincidences, return_counts=True)

            # Get features to eliminate
            features_to_eliminate = coincidences[counts == n_arms]

            # Set a maximum nmber of features to eliminate (not to pass the n_features_to_select limit)
            if len(selected_features) - len(features_to_eliminate) <= n_features_to_select:
                coincidences[:len(selected_features) - n_features_to_select]
            
            if len(features_to_eliminate) == 0:
                # print("No coincidences found, Stopping RFE at iteration", iteration)
                # break
                # Get sorted feature importance values
                mean_feature_importances = np.ndarray([len(selected_features), n_arms])
                for arm in range(n_arms):
                    mean_feature_importances[:, arm] = feature_ranking_dict[iteration][arm]["mean_feature_importances"]
                mean_feature_importances = np.mean(mean_feature_importances, axis = 1)
                # Sort feature importances
                sorted_feature_importances = np.argsort(mean_feature_importances)
                # Get least important feature to eliminate
                features_to_eliminate = feature_ranking_dict[iteration][arm]["feature_names"][sorted_feature_importances[:1]]

            print("Features to eliminate ({} features):".format(len(features_to_eliminate)), features_to_eliminate)

            # Evaluate performance with all features
            self.build_classifier(model = self.model, params = self.params, random_seed_initialization = self.random_seed_initialization, verbose = False)
            self.fit_classifiers_cv(features_to_include=selected_features, verbose = False)
            self.evaluate_classifiers_cv(make_plots = False, force_evaluation = True, verbose = False)
            rfe_line = pd.DataFrame({"Iteration": [iteration], 
                                    "N_features": [len(selected_features)], 
                                    "Train ROC": [self.results_cv["train_mean_roc_auc"]], 
                                    "Train ROC (std)": [self.results_cv["train_std_roc_auc"]],
                                    "Test ROC": [self.results_cv["test_mean_roc_auc"]],
                                    "Test ROC (std)": [self.results_cv["test_std_roc_auc"]],
                                    "Features": [selected_features]}, index = [0])
            rfe_df = pd.concat([rfe_df, rfe_line], ignore_index = True)
            if make_performance_plot:
                # Make plot with training and test ROC
                os.makedirs(self.output_dir + "/rfe", exist_ok = True)
                rfe_plot(rfe_df, self.output_dir + "/rfe")

            # Eliminate features
            selected_features = [feature for feature in selected_features if feature not in features_to_eliminate]
            iteration += 1

        print("Reached the desired number of features in", iteration, "iterations")
        print("Final number of features:", len(selected_features))
        print("Selected features:", selected_features)

        self.selected_features = selected_features

        if make_performance_plot:
            # Make plot with training and Test ROC
            rfe_plot(rfe_df)

            self.n_splits = original_n_splits
        
        if select_best:
            print("Selecting feature set with best performance")
            # Select best feature set (Get features from iteration with maximum Test ROC)
            best_iteration = rfe_df[rfe_df["Test ROC"] == rfe_df["Test ROC"].max()]
            print("Best iteration ({} features):".format(best_iteration["N_features"].values[0]), best_iteration["Iteration"].values[0])
            print("Test ROC:", best_iteration["Test ROC"].values[0])
            print("Features:", best_iteration["Features"].values[0])
            self.selected_features = best_iteration["Features"].values[0]

    def make_feature_importance_plot(self):
        plot_feature_importances(self.classifiers, self.features_to_include, self.output_dir)

    def outlier_assessment(self):
        """
        Performs outlier assessment by averaging the test predictions from all classifiers and calculating the ratio of correct
        predictions for each sample, as well as the average prediction and probability. It stores the results in the outlier_df 
        attribute and in the output_dir if it is not None.

        To be used after the evaluate_classifiers_cv method.

        """
        # Make a df with all the cases from the dataset. 
        # At each split, we will add a column with the prediction for each individual sample
        # Possible values will be either 0/1 or nan (if the sample was not in the test set for that split)
        self.outlier_df = self.original_df[[self.id_label, self.class_label]].copy()
        
        for split in range(self.n_splits):
            # Get test cases with id and class labels
            test_df = self.test_df_list[split][[self.id_label]].copy()
            # Get predictions linked to id_label for split
            test_df[f'Split {split}'] = self.results[split]['y_test_pred']
            test_df[f'Split {split} (prob)'] = self.results[split]['probs_test']
            # Merge with outlier_df
            self.outlier_df = self.outlier_df.merge(test_df, on = self.id_label, how = 'left')

        def calculate_correct_ratio(row, class_label):
            split_columns = [col for col in row.index if "Split" in col and "prob" not in col]
            n = row[split_columns].count()
            correct = sum(row[col] == row[class_label] for col in split_columns if not np.isnan(row[col]))
            return pd.Series([n, correct / n if n > 0 else np.nan], index=['N_non_nan', 'Correct_ratio'])

        # Apply the function to each row of the dataframe
        self.outlier_df[['N_non_nan', 'Correct_ratio']] = self.outlier_df.apply(lambda row: calculate_correct_ratio(row, self.class_label), axis=1)
        self.outlier_df["Average_prediction"] = self.outlier_df[[col for col in self.outlier_df.columns if "Split" in col and "prob" not in col]].mean(axis = 1)
        self.outlier_df["Average_prediction_prob"] = self.outlier_df[[col for col in self.outlier_df.columns if "Split" in col and "prob" in col]].mean(axis = 1) # Not sure this parameter really means something, all classifiers should be equally calibrated
                                                                                                                                                                  # Also, it is not clear to me what threshold should be used with the averaged probabilities
                                                                                                                                                                  # I think it may be better to stick with the average prediction

    def fine_tune(self, param_array_dict: dict = None):
        """
        Fine tunes the model by performing a grid search over the provided parameters. It stores the results in the
        fine_tuning_df and best_params attributes and in the output_dir if it is not None.

        To be used after the build_classifier method.

        Parameters
        ----------
        param_array_dict : dict
            Dictionary with the parameters to fine tune.
        
        """
        
        assert param_array_dict is not None, "Please, provide a dictionary with the parameters to fine tune"
        
        self.fine_tuning_df = None
        
        # Generate all combinations of parameters
        all_param_combinations = list(itertools.product(*param_array_dict.values()))

        for idx, param_combination in enumerate(all_param_combinations):
            # Set parameters
            params = {param: param_combination[idx_] for idx_, param in enumerate(param_array_dict.keys())}
            self.build_classifier(model = self.model, 
                                  params = params, 
                                  random_seed_initialization=self.random_seed_initialization, 
                                  verbose = False)
            
            if not os.path.exists(self.output_dir + f"/folds/{str(0)}/{self.model_name}/results.pkl"): 
                print(f"Computing cross-validation for model {self.model_name} ({1 + idx}/{len(all_param_combinations)})")
                self.fit_classifiers_cv(features_to_include=self.features_to_include, force=True, verbose=False)
                self.evaluate_classifiers_cv(make_plots = False, verbose = False)
            else:
                print(f"Loading cross-validation for model {self.model_name} ({1 + idx}/{len(all_param_combinations)})")
            self.assess_variability_cv(force=True)

            if self.fine_tuning_df is None:
                self.fine_tuning_df = self.variability_results.copy()
                # Add column for name at the beginning
                self.fine_tuning_df.insert(0, "Model", [self.model_name])
            else:
                aux_df = self.variability_results.copy()
                aux_df.insert(0, "Model", [self.model_name])
                self.fine_tuning_df = pd.concat([self.fine_tuning_df, aux_df], ignore_index=True)
        # Display sorted results by "Test ROC"[mean]
        # display(self.fine_tuning_df.sort_values(by = ("Test ROC", "mean"), ascending = False).head(10))
        print(self.fine_tuning_df.sort_values(by = ("Test ROC", "mean"), ascending = False).head(10))
        # Get parameters for best model
        best_params_list = all_param_combinations[self.fine_tuning_df.sort_values(by = ("Test ROC", "mean"), ascending = False).head(1).index.values[0]]
        self.best_params = {param: best_params_list[idx] for idx, param in enumerate(param_array_dict.keys())}
        print("Best model's parameters:")
        for key, value in self.best_params.items():
            print(f"\t{key}: {value}")

        print("\nFitting best model...")

        # Build best model
        self.build_classifier(model = self.model,
                              params = self.best_params,
                              random_seed_initialization= self.random_seed_initialization,
                              verbose = True)
        self.fit_classifiers_cv(features_to_include=self.features_to_include)
        self.evaluate_classifiers_cv(make_plots = True)
        self.assess_variability_cv()
        self.display_results()

    def assess_clinical_outcomes(self, 
                                 clinical_df: pd.DataFrame = None, 
                                 columns_to_assess: list = None):
        """
        Given a separate dataframe with a shared identification label and a list of columns to assess, this method
        merges the outlier dataframe with the clinical dataframe and computes the effect of the classification on the clinical
        data. It stores the results in the clinical_results_df attribute.

        To be used after the outlier_assessment method.

        Parameters
        ----------
        clinical_df : pd.DataFrame
            Dataframe containing the clinical data.
        columns_to_assess : list
            List of columns to assess.
            
        """
        
        assert clinical_df is not None, "Please, provide a dataframe with clinical data"
        assert columns_to_assess is not None, "Please, provide a list of columns to assess"
        assert self.id_label in clinical_df.columns, "Please, provide a dataframe with the id_label column"
        assert self.outlier_df is not None, "Please, run the outlier_assessment method first"

        self.clinical_df = clinical_df
        self.columns_to_assess = columns_to_assess

        self.clinical_df[self.id_label] = self.clinical_df[self.id_label].astype(int)
        self.clinical_df = self.clinical_df.merge(self.outlier_df, on = self.id_label, how = "left")

        row_names = ["Num 0s", "Num 1s"]
        for col in self.columns_to_assess:
            row_names.append(f"Num {col} in 0s")
            row_names.append(f"% {col} in 0s")
            row_names.append(f"Num {col} in 1s")
            row_names.append(f"% {col} in 1s")

        self.clinical_results_df = pd.DataFrame({"Attribute": row_names})

        for split in range(self.n_splits):
            split_df = self.outlier_df[self.outlier_df[f"Split {split}"].notnull()].copy()[[self.id_label, f"Split {split}"]]
            split_df[f"Split {split}"] = split_df[f"Split {split}"].astype(int)
            col_values = [split_df[f"Split {split}"].value_counts()[0],
                   split_df[f"Split {split}"].value_counts()[1]]
            # Merge with clinical_df
            split_df = split_df.merge(self.clinical_df, on = self.id_label, how = 'left')
            split_df = split_df.rename(columns = {f"Split {split}_x": f"Split {split}"})
            for col in self.columns_to_assess:
                col_values.append(split_df[split_df[f"Split {split}"] == 0][col].sum())
                col_values.append(100 * split_df[split_df[f"Split {split}"] == 0][col].sum() / split_df[split_df[f"Split {split}"] == 0][col].count())
                col_values.append(split_df[split_df[f"Split {split}"] == 1][col].sum())
                col_values.append(100 * split_df[split_df[f"Split {split}"] == 1][col].sum() / split_df[split_df[f"Split {split}"] == 1][col].count())

            self.clinical_results_df[f"Split {split}"] = col_values

        # Average results across splits
        self.clinical_results_df["Mean"] = self.clinical_results_df[[col for col in self.clinical_results_df.columns if "Split" in col]].mean(axis = 1)
        self.clinical_results_df["Std"] = self.clinical_results_df[[col for col in self.clinical_results_df.columns if "Split" in col]].std(axis = 1)