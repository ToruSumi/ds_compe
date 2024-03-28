# encoding: utf-8

import numpy as np

def evaluation_spec(y_pred) -> float:
    num = len(y_pred)
    y_pred = np.clip(y_pred, 90, 110)
    score = np.sum(np.abs(100 - y_pred)) / num
    return score