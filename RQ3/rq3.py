import os
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg
from RQ3.card import card, train_vectorizer
from utils.metrics import calc_aver
import csv


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'
headers = ('project', 'total count', 'mean jaccard', 'mean hamming',
           'mean precision', 'mean recall', 'mean f1-score', 'mean accuracy')


def start():
    vectorizer = train_vectorizer()
    jac_list = []
    ham_list = []
    prec_list = []
    rec_list = []
    f1_list = []
    acc_list = []
    with open(os.path.join(RQ3_result_dir, 'card_result.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(headers)
        for repo in repos:
            print(f'card - {repo}')
            corpus = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
            corpus = list(filter(lambda x: set(x[3]) & set(basic_types), corpus))
            jacs, hams, precs, recs, f1s, accs = card(corpus, vectorizer)
            jac_list += jacs
            ham_list += hams
            prec_list += precs
            rec_list += recs
            f1_list += f1s
            acc_list += accs
            writer.writerow((repo, len(jacs),
                             calc_aver(jacs),
                             calc_aver(hams),
                             calc_aver(precs),
                             calc_aver(recs),
                             calc_aver(f1s),
                             calc_aver(accs)))
        writer.writerow(('total', len(jac_list),
                         calc_aver(jac_list),
                         calc_aver(ham_list),
                         calc_aver(prec_list),
                         calc_aver(rec_list),
                         calc_aver(f1_list),
                         calc_aver(acc_list)))
