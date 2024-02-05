import os
os.getcwd()

import copy

from imblearn.over_sampling import RandomOverSampler, SMOTE

def preprocess_data(train_df, test_df, class_label, info_columns, oversampling = None, normalize = False):
    def min_max_norm(data):
        return (data - data.min()) / (data.max() - data.min())
    def min_max_norm_test(data, minimum, maximum):
        return (data - minimum) / (maximum - minimum)

    ignore = info_columns

    # Identify binary features
    binary_features = []
    for feature in train_df.columns:
        if feature in ignore:
            continue
        elif len(train_df[feature].unique()) == 2:
            binary_features.append(feature)

    # Preprocess train data
    preprocessed_train_df = copy.deepcopy(train_df)

    # Get min max values to normalize test data using train values
    if normalize:
        min_max_values = {}
        for feature in preprocessed_train_df.columns:
            if feature in ignore:
                continue
            min_max_values[feature] = {"min": preprocessed_train_df[feature].min(), "max": preprocessed_train_df[feature].max()}
            preprocessed_train_df[feature] = min_max_norm(preprocessed_train_df[feature])

    # Finally, we need to impute missing values. We will use the median value from the training set to impute missing values in both the training and val set
    imputation_values = {}
    for feature in preprocessed_train_df.columns:
        if feature in ignore:
            continue
        elif preprocessed_train_df[feature].isnull().sum() > 0:
            preprocessed_train_df[feature].fillna(preprocessed_train_df[feature].median(), inplace=True)
            imputation_values[feature] = preprocessed_train_df[feature].median()
        else:
            imputation_values[feature] = preprocessed_train_df[feature].median()
    
    # Preprocess test data
    preprocessed_test_df = copy.deepcopy(test_df)

    # Apply min-max normalization using the same values as the training cohort
    if normalize:
        for feature in preprocessed_test_df.columns:
            if feature in ignore:
                continue
            preprocessed_test_df[feature] = min_max_norm_test(preprocessed_test_df[feature], 
                                                            min_max_values[feature]["min"], 
                                                            min_max_values[feature]["max"])

    # Impute missing values using the same values as the training cohort
    for feature in preprocessed_test_df.columns:
        if feature in ignore:
            continue
        elif preprocessed_test_df[feature].isnull().sum() > 0:
            preprocessed_test_df[feature].fillna(imputation_values[feature], inplace=True)
            
    # Drop columns that are not used in the model
    y_train = preprocessed_train_df[class_label]
    X_train = preprocessed_train_df.drop(columns=ignore)
    if oversampling is not None:
        # Apply random oversampling to the training set
        if oversampling == "ROS":
            oversampler = RandomOverSampler(random_state=0)
        elif oversampling == "SMOTE":
            oversampler = SMOTE(random_state=0)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

    if oversampling == "SMOTE":
        # Round binary features
        for feature in binary_features:
            X_train[feature] = X_train[feature].round()
        # We checked that there weren-t any issues with categorical variables when using SMOTE, they were fine

    # Test is treated as normal
    y_test = preprocessed_test_df[class_label]
    X_test = preprocessed_test_df.drop(columns=ignore)

    for col in X_train.columns:
        for symbol in ["<", ">", "[", "]"]:
            if symbol in col:
                X_train.rename(columns={col: col.replace(symbol, "")}, inplace=True)
                X_test.rename(columns={col: col.replace(symbol, "")}, inplace=True)

    pickle_data = {"X_train": X_train, "y_train": y_train,
                   "X_test": X_test, "y_test": y_test}

    return pickle_data