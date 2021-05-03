import typing as th
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

def purity(y_pred, y_act):
    k = int(max(y_pred)) + 1
    most_classes = np.zeros(k)
    for purity_class in range(k):
        class_members = y_act[y_pred == purity_class]
        most_class = 0
        most_class_ec = sum(class_members == 0)
        for j in range(1, 8):
            temp = sum(class_members == j)
            if temp > most_class_ec:
                most_class_ec = temp
                most_class = j
        most_classes[purity_class] = most_class_ec
    return sum(most_classes) / len(y_pred)


def adjusted_rand_index(y_pred, y_act):
    return adjusted_rand_score(np.array([np.array(x) for x in y_pred]), np.array([np.array(x) for x in y_act]))


evaluation_functions = dict(purity=purity, adjusted_rand_index=adjusted_rand_index)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
