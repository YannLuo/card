import re
import csv
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import math
from collections import Counter
import os
from sklearn.feature_selection import SelectKBest, chi2


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
id2term = {}
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType']
doc_arc_dir = 'doc_archive'
RQ2_result_dir = 'RQ2_result'


def read_result(repo_name):
    with open(f'{doc_arc_dir}/{repo_name}.csv', 'r', encoding='utf-8') as rf:
        reader = csv.reader(rf)
        rl = []
        next(reader)
        for row in reader:
            rl.append(row)
    return rl


def tfidf(X, vectorizer):
    global id2term
    tot_doc_num = X.shape[0]
    include_doc_num = {}
    for k, v in vectorizer.vocabulary_.items():
        id2term[v] = k
    terms = []
    for x in X:
        terms.extend([id2term[idx] for idx in x.indices])
    terms = list(set(terms))
    for term in terms:
        term_id = vectorizer.vocabulary_[term]
        cnt = 0
        for x in X:
            if term_id in x.indices:
                cnt += 1
        include_doc_num[term_id] = cnt
    ret = []
    for x in X:
        l = []
        tot_term_num = sum(x.data)
        for i, term_id in enumerate(x.indices):
            if include_doc_num[term_id] >= max(int(X.shape[0] * 0.1 + 0.5), 2):
                term = id2term[term_id]
                item = (term, x.data[i] / tot_term_num * math.log(1.0 + (tot_doc_num + 1.0) / (include_doc_num[term_id] + 1.0)))
                l.append(item)
        ret.append(sorted(l, key=lambda x: x[1], reverse=True))
    return ret


def cal_tfidf_of_certain_type(X):
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    X = vectorizer.fit_transform(X)
    return tfidf(X, vectorizer)


def count_type_by_desc(results):
    tts = [result[2] for result in results]
    type_list = []
    for tt in tts:
        tt = tt.split('\n')
        for true_type in tt:
            tmp = re.compile('\s+').split(true_type)[1].split(',')
            type_list.extend(tmp)
    type_list = filter(lambda x: x, type_list)
    c = Counter(type_list)
    sorted_c = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return sorted_c


def extract_structed_vars_and_docstring(repo_name):
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


def extract_unstructed_vars_and_docstring_with_one_augument(repo_name):
    results = read_result(repo_name)
    var_type_doc = []
    for result in results:
        tts = result[2].split('\n')
        ds = result[-1]
        ret_idx = ds.find(':return:')
        if ret_idx != -1:
            ds = ds[:ret_idx]
        if ':param ' not in ds and len(tts) == 1:
            var_name, types = tts[0].split('  ')
            types = types.split(', ')
            res = sent_tokenize(ds)
            var_type_doc.append((result[0], result[1], var_name, types, res[0]))
    return var_type_doc


def stat_freq_used_type():
    '''
    statistic frequently-used type, use mrr
    '''
    global repos
    repo_cnt = len(repos)
    d = {}
    occur_repos = {}
    for repo_name in repos:
        results = read_result(repo_name)
        sorted_c = count_type_by_desc(results)
        for i in range(len(sorted_c)):
            d[sorted_c[i][0]] = d.get(sorted_c[i][0], 0.0) + 1.0 / (i + 1.0)
            if repo_name not in occur_repos.get(sorted_c[i][0], []):
                occur_repos.setdefault(sorted_c[i][0], []).append(repo_name)
    d_items = [(di[0], di[1] / repo_cnt) for di in d.items()]
    d_items = sorted(d_items, key=lambda x: -x[1])
    print(f'Type\tmrr\toccur_repo(s)')
    for di in d_items[:20]:
        print(f'{di[0]}\t{round(di[1], 3)}\t{len(set(occur_repos[di[0]]))}')


def stat_hint_term_by_tfidf(basic_type):
    '''
    statistic terms may imply type of variable sorted  by tfidf and mrr
    :param basic_type: str
    :return:
    '''
    mrr = {}
    X = []
    for repo in repos:
        ret = extract_structed_vars_and_docstring(repo)
        for _, _, var_name, types, docstring in ret:
            if basic_type in types:
                docstring = re.compile(':param \w+: ').split(docstring)[1]
                X.append(docstring)
    if X:
        # TODO: use chi2 and SelectKBest
        ret = cal_tfidf_of_certain_type(X)
        for doc in ret:
            for i, li in enumerate(doc):
                mrr[li[0]] = mrr.get(li[0], 0.0) + li[1]
    mrr_list = sorted(mrr.items(), key=lambda x: x[1], reverse=True)
    cur_ret = []
    for k, v in mrr_list[:10]:
        cur_ret.append((k, round(v / len(ret), 3)))
    return cur_ret


def main():
    global repos, id2term
    l = [[] for _ in range(10)]
    for basic_type in basic_types:
        ret = stat_hint_term_by_tfidf(basic_type)
        for i, r in enumerate(ret):
            l[i].append(r)
    with open(os.path.join('RQ2_result', 'key_terms.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        header = []
        for bt in basic_types:
            header.append(bt)
            header.append('')
        writer.writerow(header)
        for items in l:
            s = []
            for i, item in enumerate(items):
                s.append(item[0])
                s.append(item[1])
            writer.writerow(s)
        # for i, item in enumerate(items):
        #     s.append(f'{item[0]} & {item[1]}')
        # print(' & '.join(s) + ' \\\\')
        # for items in l:
        #     s = []
        #     for i, item in enumerate(items[:5]):
        #         s.append(f'{item[0]} & {item[1]}')
        #     print(' & '.join(s) + ' \\\\')
        # print()
        # for items in l:
        #     s = []
        #     for i, item in enumerate(items[5:]):
        #         s.append(f'{item[0]} & {item[1]}')
        #     print(' & '.join(s) + ' \\\\')


if __name__ == '__main__':
    main()
