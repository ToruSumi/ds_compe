# encoding: utf-8

import sys
import numpy as np
import pandas as pd
import optuna
import mlflow
import preprocess
import evaluation
import lightgbm as lgb
from warnings import simplefilter
from logging import info, debug
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.extend([
    str("../")
])
import common as cm

simplefilter("ignore")


def build_predict_spec(X, y, params):
    info(f"[Build LGBM] X: {X.shape} | y: {y.shape}")
    debug(f"[Build LGBM] columns: {X.columns}")
    rgr = lgb.LGBMRegressor(**params)
    rgr.fit(X, y, eval_metric=evaluation.evaluation_spec)
    return rgr


def make_spec_prediction_model(train):
    target_col = "spec"
    X = train.drop(target_col, axis=1)
    y = train[target_col]
    params = {"objective": "regression", "learning_rate": 0.05}
    rgr = build_predict_spec(X, y, params)

    return rgr


def learn_model(data, params):
    col_y = "parameter"
    X = data.drop([col_y, "spec"], axis=1)
    y = data[col_y]
    params.update({
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
    })

    lgb_train = lgb.Dataset(X, y)
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train])
    return model


def make_parameter_prediction_model(trial, train, rgr):
    target_col = "parameter"
    X = train.drop([target_col, "spec"], axis=1)
    y = train[target_col]
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 100, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20),
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000)
    }
    # mlflow.log_params(params)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    model_list = []
    fold_scores = []

    for train_index, valid_index in kf.split(X):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # LightGBM用のデータセットを作成
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

        # parameter を予測
        # モデルを学習
        model = lgb.train(
            params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_valid]
        )
        param_pred = model.predict(X_valid, num_iteration=model.best_iteration)
        debug(param_pred)
        param_diff = np.sqrt(mean_squared_error(y_valid, param_pred))
        X_valid.loc[:, "parameter"] = param_pred
        model_list.append(model)

        # specを予測してscoreを確認
        y_pred = rgr.predict(X_valid)
        fold_score = evaluation.evaluation_spec(y_pred)
        fold_scores.append([param_diff, fold_score])

    # 各foldのスコアを表示
    score_df = pd.DataFrame(fold_scores, columns=["parameter_diff", "spec_diff"])
    # info(score_df)
    # idx = score_df["spec_diff"].idxmax()
    # mlflow.log_metric("KFold_spec_diff", score_df["spec_diff"].min())
    # mlflow.lightgbm.save_model(model_list[idx], f"model/{cm.now()}")
    # return model_list[idx]
    return score_df["spec_diff"].min()


def evaluation_model(model, test, rgr):
    test1 = test.copy()
    test1["parameter"] = 0
    test1_spec = rgr.predict(test1)

    test2 = test.copy()
    test2["parameter"] = np.clip(
        model.predict(test2, num_iteration=model.best_iteration), -10, 10
    )
    info(test2["parameter"].head())
    test2_spec = rgr.predict(test2)

    info(
        f"Only 0: {evaluation.evaluation_spec(test1_spec)} | Predicted: {evaluation.evaluation_spec(test2_spec)}"
    )
    mlflow.log_metrics(
        {
            "only_0": evaluation.evaluation_spec(test1_spec),
            "predicted": evaluation.evaluation_spec(test2_spec),
        }
    )
    output_p = cm.check_directory(f"result/artifacts/{cm.now()}_prediction.csv")
    test2["parameter"].to_csv(output_p)
    mlflow.log_artifact(str(output_p))


def preprocess_data(data):
    data_pp = preprocess.add_stat_columns_each_num(data)
    data_pp = preprocess.add_polynomialfeatures(data_pp)
    data_pp = preprocess.feature_selection(data_pp)

    if "id" in data_pp.columns:
        data_pp.set_index("id", inplace=True)

    output_p = cm.check_directory(f"data/artifacts/{cm.now()}_preprocessed.csv")
    data_pp.to_csv(output_p)
    mlflow.log_artifact(str(output_p))
    return data_pp

def read_data():
    data1 = pd.read_csv("input/data_1.csv")
    data2 = pd.read_csv("input/data_2.csv")
    train = pd.concat([data1, data2], ignore_index=True)
    test = pd.read_csv("input/test.csv")

    return train, test

def prediction():
    mlflow.start_run(experiment_id="0")
    train, test = read_data()
    train = preprocess_data(train)
    rgr = make_spec_prediction_model(train)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: make_parameter_prediction_model(
            trial=trial, train=train, rgr=rgr
        ),
        n_trials=20,
    )
    model = learn_model(train, study.best_params)
    mlflow.log_params(study.best_params)
    test = preprocess_data(test)
    evaluation_model(model, test, rgr)

    mlflow.log_artifact("predict.py")
    mlflow.log_artifact("../common.py")
    mlflow.log_artifact("preprocess.py")
    mlflow.log_artifact("evaluation.py")
    mlflow.end_run()


if __name__ == "__main__":
    cm.set_log("log/predict.log", "info")
    prediction()
