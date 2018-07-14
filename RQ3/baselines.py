from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from utils.metric import jaccard
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg
import os
import csv


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'


def gen_corpus_data(vectorizer, types_list, sep_docstrings):
    term_len = len(vectorizer.vocabulary_.items())
    type_len = len(basic_types)
    X = []
    y = []
    for types, sds in zip(types_list, sep_docstrings):
        term_vec = [0] * term_len
        type_vec = [0] * type_len
        union = set(types) & set(basic_types)
        if union:
            for idx, cnt in zip(sds.indices, sds.data):
                term_vec[idx] = cnt
            for t in types:
                if t in basic_types:
                    type_vec[basic_types.index(t)] = 1
            X.append(term_vec)
            y.append(type_vec)
    return X, y


def train_model(clf_type='rf'):
    clf_type = clf_type.lower()
    docstrings = []
    types_list = []
    for repo in repos:
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        types_list.extend([r[-2] for r in result])
        docstrings.extend([r[-1] for r in result])
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    sep_docstrings = vectorizer.fit_transform(docstrings)
    X, y = gen_corpus_data(vectorizer, types_list, sep_docstrings)
    if clf_type == 'rf':
        clf = RandomForestClassifier().fit(X, y)
    elif clf_type == 'knn':
        clf = KNeighborsClassifier().fit(X, y)
    else:
        return None, None
    return clf, vectorizer


def compare(clf_type):
    clf_type = clf_type.lower()
    model, vectorizer = train_model(clf_type)
    jac_list = []
    with open(os.path.join(RQ3_result_dir, f'{clf_type}_result.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        for repo in repos:
            print(repo)
            docstrings = []
            types_list = []
            result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
            types_list.extend([r[-2] for r in result])
            docstrings.extend([r[-1] for r in result])
            sep_docstrings = vectorizer.transform(docstrings)
            X, y = gen_corpus_data(vectorizer, types_list, sep_docstrings)
            tmp_jac_list = []
            if X:
                pred_result = model.predict(X)
                for idx, pr in enumerate(pred_result):
                    true_types_set = set(types_list[idx]) & set(basic_types)
                    pred_types = []
                    for i, x in enumerate(pr):
                        if x > 0.20:
                            pred_types.append(basic_types[i])
                    pred_types_set = set(pred_types)
                    if 'Tuple' in pred_types_set:
                        pred_types_set -= set(['Tuple'])
                        pred_types_set.add('List')
                    if 'Tuple' in true_types_set:
                        true_types_set -= set(['Tuple'])
                        true_types_set.add('List')
                    tmp_jac = jaccard(pred_types_set, true_types_set)
                    tmp_jac_list.append(tmp_jac)
                    jac_list.append(tmp_jac)
            tmp_cons_cnt = sum([jac >= 1.0 for jac in tmp_jac_list])
            # print(repo, len(tmp_jac_list), '&', tmp_cons_cnt, '&', round(sum(tmp_jac_list) / len(tmp_jac_list), 3))

            # print(f'Mean jaccard: {0.0 if len(tmp_jac_list) == 0 else sum(tmp_jac_list) / len(tmp_jac_list):.3f} '
            #       f'consistent count: {tmp_cons_cnt} total count: {len(tmp_jac_list)}')
            writer.writerow((repo,
                             len(tmp_jac_list),
                             tmp_cons_cnt,
                             round(0.0 if len(tmp_jac_list) == 0 else sum(tmp_jac_list) / len(tmp_jac_list), 3)))
        writer.writerow(('total',
                         len(jac_list),
                         sum([jac >= 1.0 for jac in jac_list]),
                         round(0.0 if len(jac_list) == 0 else sum(jac_list) / len(jac_list), 3)))
        # print('RF')
        # print(f'Accuracy: {sum([jac >= 1.0 for jac in jac_list])}/{len(jac_list)}')
        # print(f'Mean jaccard: {sum(jac_list) / len(jac_list)}')
