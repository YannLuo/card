def jaccard(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def hamming(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    return len(s1 ^ s2)


def precision(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    return 0.0 if len(s1) == 0 else len(s1 & s2) / len(s1)


def recall(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    return len(s1 & s2) / len(s2)


def f1_score(l1, l2):
    prec = precision(l1, l2)
    rec = recall(l1, l2)
    return 0.0 if prec + rec < 1e-3 else 2 * prec * rec / (prec + rec)


def accuracy(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    return s1 == s2


def calc_aver(l):
    return round(0.0 if len(l) == 0 else sum(l) / len(l), 3)
