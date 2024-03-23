# encoding: utf-8

import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    tune_model,
    evaluate_model,
    stack_models,
)
import argparse
from pathlib import Path
from tqdm.auto import tqdm


def evaluation(train: pd.DataFrame, target: str):
    exp = setup(
        data=train,
        target=target,
        train_size=0.7,
        normalize=True,
        polynomial_features=True,
        feature_selection=True,
        fold_shuffle=True,
        use_gpu=True,
        log_experiment=True,
    )
    # 比較するモデルを選択する
    model_top3 = compare_models(fold=5, n_select=3)

    # 最良のモデルをチューニングする
    tuned_models = list()
    for model in tqdm(model_top3):
        tuned_models.append(
            tune_model(
                model, search_library="scikit-optimize", search_algorithm="bayesian"
            )
        )

    # モデルをスタックする
    stacker = stack_models(tuned_models)

    # モデルを評価する
    evaluate_model(stacker)


def read_data(train: str, test: str):
    train = Path(train).resolve()
    train_df = pd.read_csv(train)

    test = Path(test).resolve()
    test_df = pd.read_csv(test)

    return train_df, test_df


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--train",
        type=str,
        help="Train data path",
        dest="train",
    )
    argparser.add_argument(
        "--test",
        type=str,
        help="Test data path",
        dest="test",
    )
    argparser.add_argument(
        "--target",
        type=str,
        help="Target column name",
        dest="target",
    )
    args = argparser.parse_args()
    train, test = read_data(train=args.train, test=args.test)
    evaluation(train=train, target=args.target)


if __name__ == "__main__":
    main()
