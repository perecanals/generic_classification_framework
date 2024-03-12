import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

def split_data(raw_df, class_label, ratio, random_seed, verbose=False):
    np.random.seed(random_seed)

    # Get unique values of the class label
    unique_values = raw_df[class_label].astype(int).unique()

    if verbose:
        print("Total number of samples: ", len(raw_df))
        for value in unique_values:
            print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(raw_df[raw_df[class_label] == value]), 100 * len(raw_df[raw_df[class_label] == value]) / len(raw_df)))

    # Split into train and test (stratified)
    train_df, test_df = train_test_split(raw_df, test_size=ratio, random_state=random_seed, stratify=raw_df[class_label])

    if verbose:
        print("Train:")
        print("Total number of samples: ", len(train_df))
        for value in unique_values:
            print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(train_df[train_df[class_label] == value]), 100 * len(train_df[train_df[class_label] == value]) / len(train_df)))

        print("Test:")
        print("Total number of samples: ", len(test_df))
        for value in unique_values:
            print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(test_df[test_df[class_label] == value]), 100 * len(test_df[test_df[class_label] == value]) / len(test_df)))

    return train_df, test_df

def split_data_kfold(raw_df, class_label, k, random_seed, verbose=False):
    np.random.seed(random_seed)

    # Get unique values of the class label
    unique_values = raw_df[class_label].astype(int).unique()

    if verbose:
        print("Total number of samples: ", len(raw_df))
        for value in unique_values:
            print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(raw_df[raw_df[class_label] == value]), 100 * len(raw_df[raw_df[class_label] == value]) / len(raw_df)))

    # Split into train and test (stratified)
    skf = StratifiedKFold(n_splits=k, random_state=random_seed, shuffle=True)

    train_dfs = []
    test_dfs = []
    for train_index, test_index in skf.split(raw_df, raw_df[class_label]):
        train_df, test_df = raw_df.iloc[train_index], raw_df.iloc[test_index]

        if verbose:
            print("Train:")
            print("Total number of samples: ", len(train_df))
            for value in unique_values:
                print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(train_df[train_df[class_label] == value]), 100 * len(train_df[train_df[class_label] == value]) / len(train_df)))

            print("Test:")
            print("Total number of samples: ", len(test_df))
            for value in unique_values:
                print("Total number of samples from class {}: {} ({:.2f}%)".format(value, len(test_df[test_df[class_label] == value]), 100 * len(test_df[test_df[class_label] == value]) / len(test_df)))

        train_dfs.append(train_df)
        test_dfs.append(test_df)

    return train_dfs, test_dfs