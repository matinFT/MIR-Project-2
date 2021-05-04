import typing as th


def accuracy(y, y_hat) -> float:
    matches = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            matches += 1
    return matches / len(y)


def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    p1, p0 = precision(y, y_hat)
    r1, r0 = recall(y, y_hat)
    return  2 * p0 * r0 / (p0 + r0), 2 * p1 * r1 / (p1 + r1) 
#     return (beta**2 + 1) * p * r / (b**2 * p + r)


def precision(y, y_hat) -> float:
    pos_matches = 0
    pos_total = 0
    neg_matches = 0
    neg_total = 0
    for i in range(len(y)):
        if y_hat[i] == 1:
            pos_total += 1
            if y[i] == 1:
                pos_matches += 1
        else:
            neg_total += 1
            if y[i] == 0:
                neg_matches += 1
                
    return neg_matches / neg_total, pos_matches / pos_total


def recall(y, y_hat) -> float:
    pos_matches = 0
    total_pos = 0
    neg_matches = 0
    total_neg = 0
    for i in range(len(y)):
        if y[i] == 1:
            total_pos += 1
            if y_hat[i] == 1:
                pos_matches += 1
        else:
            total_neg += 1
            if y_hat[i] == 0:
                neg_matches += 1
                
    return neg_matches / total_neg, pos_matches / total_pos


evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
