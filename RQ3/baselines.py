from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from utils.metrics import jaccard, hamming, accuracy, precision, recall, f1_score, calc_aver
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg
from RQ3.card import train_vectorizer
import os
import csv


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'
headers = ('project', 'total count', 'mean jaccard', 'mean hamming',
           'mean precision', 'mean recall', 'mean f1-score', 'mean accuracy')


def gen_corpus_data(vectorizer, types_list, sep_docstrings):
    term_len = len(vectorizer.vocabulary_.items())
    type_len = len(basic_types)
    X = []
    y = []
    for types, sds in zip(types_list, sep_docstrings):
        term_vec = [0] * term_len
        type_vec = [0] * type_len
        if set(types) & set(basic_types):
            for idx, cnt in zip(sds.indices, sds.data):
                term_vec[idx] = cnt
            for t in types:
                if t in basic_types:
                    type_vec[basic_types.index(t)] = 1
            X.append(term_vec)
            y.append(type_vec)
    return X, y


def train_model(clf_type, vectorizer, target=None):
    clf_type = clf_type.lower()
    docstrings = []
    types_list = []
    for repo in repos:
        if repo == target:
            continue
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        types_list.extend([r[-2] for r in result])
        docstrings.extend([r[-1] for r in result])
    sep_docstrings = vectorizer.transform(docstrings)
    X, y = gen_corpus_data(vectorizer, types_list, sep_docstrings)
    if clf_type == 'rf':
        clf = RandomForestClassifier().fit(X, y)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier().fit(X, y)
    else:
        return None, None
    return clf


def calc_metric(pred, true):
    tmp_jac = jaccard(pred, true)
    tmp_ham = hamming(pred, true)
    tmp_prec = precision(pred, true)
    tmp_rec = recall(pred, true)
    tmp_f1 = f1_score(pred, true)
    tmp_acc = accuracy(pred, true)
    return tmp_jac, tmp_ham, tmp_prec, tmp_rec, tmp_f1, tmp_acc


def compare(clf_type):
    clf_type = clf_type.lower()
    vectorizer = train_vectorizer()
    jac_list = []
    ham_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    acc_list = []
    with open(os.path.join(RQ3_result_dir, f'{clf_type}_result.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(headers)
        for repo in repos:
            model = train_model(clf_type, vectorizer, repo)
            print(f'{clf_type} - {repo}')
            docstrings = []
            types_list = []
            result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
            types_list.extend([r[-2] for r in result if set(r[-2]) & set(basic_types)])
            docstrings.extend([r[-1] for r in result if set(r[-2]) & set(basic_types)])
            sep_docstrings = vectorizer.transform(docstrings)
            X, y = gen_corpus_data(vectorizer, types_list, sep_docstrings)
            tmp_jac_list = []
            tmp_ham_list = []
            tmp_prec_list = []
            tmp_rec_list = []
            tmp_f1_list = []
            tmp_acc_list = []
            if X:
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
            writer.writerow((repo, len(tmp_jac_list),
                             calc_aver(tmp_jac_list),
                             calc_aver(tmp_ham_list),
                             calc_aver(tmp_prec_list),
                             calc_aver(tmp_rec_list),
                             calc_aver(tmp_f1_list),
                             calc_aver(tmp_acc_list)))
        writer.writerow(('total', len(jac_list),
                         calc_aver(jac_list),
                         calc_aver(ham_list),
                         calc_aver(prec_list),
                         calc_aver(rec_list),
                         calc_aver(f1_list),
                         calc_aver(acc_list)))
