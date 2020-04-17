import numpy as np
import pandas as pd

def simple_feat_scale(df, column_name):
    return df[column_name]/df[column_name].max()


def min_max_scale(df, column_name):
    return (df[column_name]-df[column_name].min())/(df[column_name].max()-df[column_name].min())


def z_score_scale(df, column_name):
    return (df[column_name]-df[column_name].mean())/df[column_name].std()

def bin_column(df, column_name, n_bins = 0, group_names = None):
    bins = np.linspace(min(df[column_name]), max(df[column_name]), n_bins + 1)
    if(group_names is None):
        return pd.cut(df[column_name], bins, include_lowest = True)
    else:
        n_bins = len(group_names)
        return pd.cut(df[column_name], bins, labels = group_names, include_lowest = True)


\