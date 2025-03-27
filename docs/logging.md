- [実験管理](#実験管理)
  - [MLflow](#mlflow)
    - [実験作成](#実験作成)
    - [パラメーターの記録](#パラメーターの記録)
    - [ファイルの保存](#ファイルの保存)
    - [性能指標の記録](#性能指標の記録)

# 実験管理

様々な前処理、モデルを構築したうえで、どの結果がベストかを比較するために、実験管理は重要になってくる。

## MLflow

[MLflow](https://mlflow.org/)は機械学習の実験管理を行うためのツールで、以下の機能を提供している。

- パラメータの管理
- メトリクスの管理
- モデルの管理

自動的にロギングする機能もあるが、以下では手動で記録する場合によく使うものをまとめる。

### 実験作成

実験に名前をつけて作成する
```python
mlflow.set_experiment("experiment_name")
```

実験名から実験IDを取得して実験管理を開始する
```python
mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("experiment_name").experiment_id)
```

### パラメーターの記録

辞書型のパラメーターをまとめて保存する
```python
mlflow.log_params(params)
```

一つだけの場合は以下のように記録する
```python
mlflow.log_param("param_name", param)
```

### ファイルの保存

事前にファイルを保存しておき、それをMLflowに保存する
```python
mlflow.log_artifact("path/to/file")
```

例えば、DataFrameをCSVファイルとして保存して、それをMLflowに記録するなら：
```python
# DataFrameをCSVとして保存
import pandas as pd

# 適当なデータでDataFrameを作成
data = {
    'col1': [1, 2, 3, 4, 5],
    'col2': ['A', 'B', 'C', 'D', 'E'],
    'col3': [0.1, 0.2, 0.3, 0.4, 0.5]
}
df = pd.DataFrame(data)

# CSVとして保存
csv_path = "data.csv"
df.to_csv(csv_path, index=False)

# MLflowに記録
mlflow.log_artifact(csv_path)
```

### 性能指標の記録

性能指標を記録する場合は以下のように記録する
```python
mlflow.log_metrics(metrics)
```

一つだけの場合は以下のように記録する
```python
mlflow.log_metric("metric_name", metric)
```

例えば、AccuracyとF1スコアを記録するなら：
```python
from sklearn.metrics import accuracy_score, f1_score

# 予測結果と正解ラベルを用意
y_true = [0, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0]

# AccuracyとF1スコアを計算
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# MLflowに記録
mlflow.log_metrics({"accuracy": accuracy, "f1": f1})
```