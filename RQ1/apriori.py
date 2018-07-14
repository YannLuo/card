"""
Description     : Simple Python implementation of the Apriori Algorithm

Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence

    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""

import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def return_items_with_min_support(item_set, transaction_list, min_support, freq_set):
    """calculates the support for items in the itemSet and returns a subset
   of the itemSet each of whose elements satisfies the minimum support"""
    _item_set = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in transaction_list:
            if item.issubset(transaction):
                freq_set[item] += 1
                local_set[item] += 1

    for item, count in list(local_set.items()):
        support = float(count)/len(transaction_list)

        if support >= min_support:
            _item_set.add(item)

    return _item_set


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in item_set for j in item_set if len(i.union(j)) == length])


def get_item_set_transaction_list(data_iterator):
    transaction_list = list()
    item_set = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transaction_list.append(transaction)
        for item in transaction:
            item_set.add(frozenset([item]))              # Generate 1-itemSets
    return item_set, transaction_list


def run_apriori(data_iter, min_support, min_confidence):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    item_set, transaction_list = get_item_set_transaction_list(data_iter)

    freq_set = defaultdict(int)
    large_set = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    one_c_set = return_items_with_min_support(item_set,
                                              transaction_list,
                                              min_support,
                                              freq_set)

    current_l_set = one_c_set
    k = 2
    while(current_l_set != set([])):
        large_set[k-1] = current_l_set
        current_l_set = join_set(current_l_set, k)
        current_c_set = return_items_with_min_support(current_l_set,
                                                      transaction_list,
                                                      min_support,
                                                      freq_set)
        current_l_set = current_c_set
        k = k + 1

    def get_support(item):
        """local function which Returns the support of an item"""
        return float(freq_set[item])/len(transaction_list)

    to_ret_items = []
    for key, value in list(large_set.items()):
        to_ret_items.extend([(tuple(item), get_support(item))
                             for item in value])

    to_ret_rules = []
    for key, value in list(large_set.items())[1:]:
        for item in value:
            _subsets = list(map(frozenset, [x for x in subsets(item)]))
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = get_support(item)/get_support(element)
                    if confidence >= min_confidence:
                        to_ret_rules.append(((tuple(element), tuple(remain)),
                                             confidence))
    return to_ret_items, to_ret_rules


def print_results(items, rules):
    """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
    for item, support in sorted(items, key=lambda item_support: item_support[1]):
        print("item: %s , %.3f" % (str(item), support))
    print("\n------------------------ RULES:")
    for rule, confidence in sorted(rules, key=lambda rule_confidence: rule_confidence[1]):
        pre, post = rule
        print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


def data_from_file(fname):
    """Function which reads from the file and yields a generator"""
    file_iter = open(fname, 'r', encoding='utf-8')
    for line in file_iter:
        line = line.strip().rstrip(',')                         # Remove trailing comma
        record = frozenset(line.split(','))
        yield record


if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='min_s',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='min_c',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    in_file = None
    if options.input is None:
        in_file = sys.stdin
    elif options.input is not None:
        in_file = data_from_file(options.input)
    else:
        print('No dataset filename specified, system with exit\n')
        sys.exit('System will exit')

    min_support = options.min_s
    min_confidence = options.min_c

    items, rules = run_apriori(in_file, min_support, min_confidence)

    print_results(items, rules)
