def jaccard(l1, l2):
    s1 = set(l1)
    s2 = set(l2)
    if len(s1 | s2) == 0:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)
