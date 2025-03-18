# encoding: utf-8

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import mlflow
from logging import info

usecols = None

def add_polynomialfeatures(data, store_mlflow=True):
    features = list(set(data.columns) ^ (set(["id", "spec", "parameter"]) & set(data.columns)))
    others = list(set(["id", "spec", "parameter"]) & set(data.columns))

    pf = PolynomialFeatures(degree=2, interaction_only=True)
    poly_df = pd.DataFrame(
        pf.fit_transform(data[features]), columns=pf.get_feature_names_out(features)
    ).drop("1", axis=1)
    others_df = data[others]

    data_pp = pd.concat([poly_df, others_df], axis=1)
    if store_mlflow:
        mlflow.log_params(
            {
                "PolynomialFeatures": True,
                "PolynomialFeatures_columns_before": len(features),
                "PolynomialFeatures_columns_after": len(poly_df.columns),
            }
        )

    info(f"[Polynomial]{data_pp.head()}")
    return data_pp

def add_stat_columns_each_num(data, store_mlflow=True):
    data_pp = data.copy()
    for i in range(1, 6):
        data_pp[f"obs_x{i}_mean"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].mean(axis=1)
        data_pp[f"obs_x{i}_median"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].median(axis=1)
        data_pp[f"obs_x{i}_std"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].std(axis=1)
        data_pp[f"obs_x{i}_min"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].min(axis=1)
        data_pp[f"obs_x{i}_max"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].max(axis=1)
        data_pp[f"obs_x{i}_skew"] = data_pp[[f"obs_{j}{i}" for j in range(1, 4)]].skew(axis=1)

    if store_mlflow:
        old = list(set(data.columns) ^ (set(["id", "spec", "parameter"]) & set(data.columns)))
        new = list(set(data_pp.columns) ^ (set(["id", "spec", "parameter"]) & set(data_pp.columns)))
        mlflow.log_params(
            {
                "AddStats": True,
                "AddStats_columns_before": len(old),
                "AddStats_after": len(new)
            }
        )

    info(f"[Stat]{data_pp.head()}")
    return data_pp

def feature_selection(data, store_mlflow=True):
    global usecols
    if usecols is not None:
        cols = list(set(usecols) & set(data.columns))
        data_pp = data[cols].copy()
    else:
        # others = list(set(["spec", "parameter"]) & set(data.columns))
        # features = list(set(data.columns) ^ set(others))

        corr = data.corr()
        corr = abs(corr["spec"]).drop(["parameter", "id"])

        cols = corr[corr > 0.1].index.to_list()

        usecols = pd.Series(cols + ["spec", "parameter", "id"]).drop_duplicates().tolist()
        data_pp = data[usecols].copy()

        if store_mlflow:
            mlflow.log_params({
                "FeatureSelection": True,
                "FS_method": "Correlation",
                "FS_before": len(data.columns),
                "FS_after": len(data_pp.columns)
            })

    info(f"[Feature Selection]{data_pp.head()}")
    return data_pp