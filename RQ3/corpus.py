import re
from nltk import sent_tokenize
import csv
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'


def gen_corpus_data(types_list, sep_docstrings):
    vectorizer = train_vectorizer()
    term_len = len(vectorizer.vocabulary_.items())
    type_len = len(basic_types)
    X = []
    y = []
    for types, sds in zip(types_list, sep_docstrings):
        term_vec = [0] * term_len
        type_vec = [0] * type_len
        for idx, cnt in zip(sds.indices, sds.data):
            term_vec[idx] = cnt
        for t in types:
            if t in basic_types:
                type_vec[basic_types.index(t)] = 1
        X.append(term_vec)
        y.append(type_vec)
    return np.array(X), np.array(y)


def read_result(repo_name):
    with open(f'doc_archive/{repo_name}.csv', 'r', encoding='utf-8') as rf:
        reader = csv.reader(rf)
        rl = []
        next(reader)
        for row in reader:
            rl.append(row)
    return rl


def ext_structed_docs(repo_name):
    results = read_result(repo_name)
    var_type_doc = []
    for result in results:
        tts = result[2].split('\n')
        ds = result[-1]
        ret_idx = ds.find(':return:')
        if ret_idx != -1:
            ds = ds[:ret_idx]
        ds = ' '.join(re.compile('\s+').split(ds))
        ds = re.compile('(:param \w+:)').split(ds)[1:]
        ds = [ds[i] + ds[i + 1] for i in range(0, len(ds), 2) if ds[i + 1].strip()]
        if ds:
            for tt in tts:
                var_name, types = tt.split('  ')
                types = types.split(', ')
                for describe in ds:
                    if describe.startswith(f':param {var_name}:'):
                        var_type_doc.append((result[0], result[1], var_name, types, describe))
    return var_type_doc


def ext_unstructed_docs_with_one_arg(repo_name):
    results = read_result(repo_name)
    var_type_doc = []
    for result in results:
        tts = result[2].split('\n')
        ds = result[-1]
        if all([x not in ds.lower() for x in ['return', ':param']]) \
                and len(tts) == 1:
            ret_idx = ds.find(':return:')
            if ret_idx != -1:
                ds = ds[:ret_idx]
            var_name, types = tts[0].split('  ')
            types = types.split(', ')
            res = sent_tokenize(ds)
            var_type_doc.append((result[0], result[1], var_name, types, res[0]))
    return var_type_doc


def train_vectorizer():
    docstrings = []
    for repo in repos:
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        docstrings.extend([r[-1] for r in result])
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english')).fit(docstrings)
    return vectorizer


def feature_selector(k=10):
    vectorizer = train_vectorizer()
    docstrings = []
    types_list = []
    for repo in repos:
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        result = list(filter(lambda x: set(x[3]) & set(basic_types), result))
        types_list.extend([r[-2] for r in result])
        docstrings.extend([r[-1] for r in result])
    sep_docstrings = vectorizer.transform(docstrings)
    X, y = gen_corpus_data(types_list, sep_docstrings)
    skb = SelectKBest(score_func=chi2, k=k).fit(X, y)
    return skb
