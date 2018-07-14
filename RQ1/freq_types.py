import csv
import os
from collections import Counter
import json
from utils.sql import SqliteOperator
import time

doc_arc_dir = 'doc_archive'
doc_sql_dir = 'doc_sql'
trace_dir = 'trace_sql'
repos = ['asphalt', 'bs4', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh', 'faker']
RQ1_result_dir = 'RQ1_result'


def stat_freq_used_type():
    repo_cnt = len(repos)
    d = {}
    occur_repos = {}
    for repo_name in repos:
        stt = time.clock()
        sorted_c = count_type_by_desc(repo_name)
        for i in range(len(sorted_c)):
            d[sorted_c[i][0]] = d.get(sorted_c[i][0], 0.0) + 1.0 / (i + 1.0)
            if repo_name not in occur_repos.get(sorted_c[i][0], []):
                occur_repos.setdefault(sorted_c[i][0], []).append(repo_name)
        edt = time.clock()
        print(f'Process {repo_name} cost {edt - stt:.3} seconds.')
    d_items = [(di[0], di[1] / repo_cnt, len(set(occur_repos[di[0]]))) for di in d.items()]
    d_items = sorted(d_items, key=lambda x: -x[-1])
    with open(os.path.join(RQ1_result_dir, 'freq_used_types.csv'), 'w', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(('Type', 'MRR', 'occur_repo(s)'))
        for di in d_items[:20]:
            writer.writerow((di[0], round(di[1], 3), len(set(occur_repos[di[0]]))))
            # print(f'{di[0]} & {round(di[1], 3)} & {len(set(occur_repos[di[0]]))} \\\\')


def count_type_by_desc(repo_name):
    print(f'Dumping {repo_name}...')
    trace_path = os.path.join(trace_dir, '%s.sqlite3' % (repo_name,))
    op_ttype = SqliteOperator(trace_path)
    sql = 'select arg_types from monkeytype_call_traces '
    result = op_ttype.select(sql)
    type_list = []
    for r in result:
        j = json.loads(r[0])
        for ji in j.items():
            type_list.append(ji[1]['qualname'])
    c = Counter(type_list)
    sorted_c = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return sorted_c


def main():
    stat_freq_used_type()


if __name__ == '__main__':
    main()
