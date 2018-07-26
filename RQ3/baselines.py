from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from utils.metrics import jaccard, hamming, accuracy, precision, recall, f1_score, calc_aver
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg, train_vectorizer, gen_corpus_data, feature_selector
import os
import csv
from tqdm import tqdm


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'
headers = ('project', 'total count', 'mean jaccard', 'mean hamming',
           'mean precision', 'mean recall', 'mean f1-score', 'mean accuracy')
vectorizer = train_vectorizer()
feat_num = 1197
prop = 0.2
selector = feature_selector(int(feat_num * prop))


def train_model(clf_type, target=None):
    global vectorizer
    clf_type = clf_type.lower()
    docstrings = []
    types_list = []
    for repo in repos:
        if repo == target:
            continue
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        result = list(filter(lambda x: set(x[3]) & set(basic_types), result))
        types_list.extend([r[-2] for r in result])
        docstrings.extend([r[-1] for r in result])
    sep_docstrings = vectorizer.transform(docstrings)
    X, y = gen_corpus_data(types_list, sep_docstrings)
    X = selector.transform(X)
    if clf_type == 'rf':
        clf = RandomForestClassifier().fit(X, y)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier().fit(X, y)
    elif clf_type == 'dt':
        clf = DecisionTreeClassifier().fit(X, y)
    elif clf_type == 'mlp':
        clf = MLPClassifier().fit(X, y)
    else:
        return None
    return clf


def calc_metric(pred, true):
    tmp_jac = jaccard(pred, true)
    tmp_ham = hamming(pred, true)
    tmp_prec = precision(pred, true)
    tmp_rec = recall(pred, true)
    tmp_f1 = f1_score(pred, true)
    tmp_acc = accuracy(pred, true)
    return tmp_jac, tmp_ham, tmp_prec, tmp_rec, tmp_f1, tmp_acc


def baseline(clf_type):
    global vectorizer
    clf_type = clf_type.lower()
    jac_list = []
    ham_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    acc_list = []
    epoch = 100
    with open(os.path.join(RQ3_result_dir, f'{clf_type}_result.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(headers)
        for _ in tqdm(range(epoch)):
            for repo in repos:
                # print(f'{clf_type} - {repo}')
                model = train_model(clf_type, target=repo)
                corpus = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
                corpus = list(filter(lambda x: set(x[3]) & set(basic_types), corpus))
                types_list = [r[-2] for r in corpus]
                docstrings = [r[-1] for r in corpus]
                if docstrings:
                    sep_docstrings = vectorizer.transform(docstrings)
                    X, y = gen_corpus_data(types_list, sep_docstrings)
                    X = selector.transform(X)
                    tmp_jac_list = []
                    tmp_ham_list = []
                    tmp_prec_list = []
                    tmp_rec_list = []
                    tmp_f1_list = []
                    tmp_acc_list = []
                    pred_result = model.predict(X)
                    for idx, pr in enumerate(pred_result):
                        true_types_set = set(types_list[idx]) - set(['None'])
                        pred_types_set = set([basic_types[i] for i, x in enumerate(pr) if x == 1])
                        if 'Tuple' in pred_types_set:
                            pred_types_set -= set(['Tuple'])
                            pred_types_set.add('List')
                        if 'Tuple' in true_types_set:
                            true_types_set -= set(['Tuple'])
                            true_types_set.add('List')
                        tmp_jac, tmp_ham, tmp_prec, tmp_rec, tmp_f1, tmp_acc = calc_metric(pred_types_set, true_types_set)
                        tmp_jac_list.append(tmp_jac)
                        tmp_ham_list.append(tmp_ham)
                        tmp_prec_list.append(tmp_prec)
                        tmp_rec_list.append(tmp_rec)
                        tmp_f1_list.append(tmp_f1)
                        tmp_acc_list.append(tmp_acc)
                    jac_list += tmp_jac_list
                    ham_list += tmp_ham_list
                    prec_list += tmp_prec_list
                    rec_list += tmp_rec_list
                    f1_list += tmp_f1_list
                    acc_list += tmp_acc_list
                # writer.writerow((repo, len(tmp_jac_list),
                #                  calc_aver(tmp_jac_list),
                #                  calc_aver(tmp_ham_list),
                #                  calc_aver(tmp_prec_list),
                #                  calc_aver(tmp_rec_list),
                #                  calc_aver(tmp_f1_list),
                #                  calc_aver(tmp_acc_list)))
        writer.writerow(('total', int(len(jac_list) / epoch),
                         calc_aver(jac_list),
                         calc_aver(ham_list),
                         calc_aver(prec_list),
                         calc_aver(rec_list),
                         calc_aver(f1_list),
                         calc_aver(acc_list)))
