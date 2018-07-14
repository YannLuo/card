import os
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg
from RQ3.card import card
import csv


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'


def start(vectorizer):
    all_corpus = 0
    all_cons = 0
    all_exp = 0
    jac_list = []
    with open(os.path.join(RQ3_result_dir, 'card_result.csv'), 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(('project', 'total count', 'consistent count', 'mean jaccard'))
        for repo in repos:
            print(repo)
            result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
            all_corpus += len(result)
            jacs = card(result, vectorizer)
            cons = sum([jac>=1.0 for jac in jacs])
            total = len(jacs)
            all_cons += cons
            all_exp += total
            jac_list.extend(jacs)
            writer.writerow((repo,
                             len(jacs),
                             sum([jac >= 1.0 for jac in jacs]),
                             round(0.0 if len(jacs) == 0 else sum(jacs) / len(jacs), 3)))
        writer.writerow(('total',
                         len(jac_list),
                         sum([jac >= 1.0 for jac in jac_list]),
                         round(0.0 if len(jac_list) == 0 else sum(jac_list) / len(jac_list), 3)))
    # print(f'Total count of corpus is {all_corpus}. {all_exp} are at least with one basic type. '
    #       f'{all_corpus - all_exp} only contain user-defined type(s).')
    # print(f'Accuracy: {all_cons}/{all_exp}')
    # print(f'Mean jaccard: {sum(jac_list) / len(jac_list)}')
