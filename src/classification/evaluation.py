import typing as th


def accuracy(y, y_hat) -> float:
    matches = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            matches += 1
    return matches / len(y)


def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    p = precision(y, y_hat)
    r = recall(y, y_hat)
    return 2 * p * r / (p + r)
#     return (beta**2 + 1) * p * r / (b**2 * p + r)


def precision(y, y_hat) -> float:
    matches = 0
    total = 0
    for i in range(len(y)):
        if y_hat[i] == 1:
            total += 1
            if y[i] == 1:
                matches += 1
    return matches / total


def recall(y, y_hat) -> float:
    matches = 0
    total = 0
    for i in range(len(y)):
        if y[i] == 1:
            total += 1
            if y_hat[i] == 1:
                matches += 1
    return matches / total


evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
