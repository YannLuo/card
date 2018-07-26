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


def read_result(repo_name):
    with open(f'{doc_arc_dir}/{repo_name}.csv', 'r', encoding='utf-8') as rf:
        reader = csv.reader(rf)
        rl = []
        next(reader)
        for row in reader:
            rl.append(row)
    return rl


def count_type_by_desc(results):
    tts = [result[2] for result in results]
    type_list = []
    for tt in tts:
        tt = tt.split('\n')
        for true_type in tt:
            varname, types = true_type.split(' ' * 2)
            tmp = [t.strip() for t in types.split(',')]
            type_list.extend(tmp)
    type_list = filter(None, type_list)
    c = Counter(type_list)
    sorted_c = sorted(c.items(), key=lambda x: x[1], reverse=True)
    return sorted_c


def stat_freq_used_type():
    repo_cnt = len(repos)
    d = {}
    occur_repos = {}
    for repo_name in repos:
        # results = read_result(repo_name)
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
    with open('freq_used_types.csv', 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        writer.writerow(('Type', 'MRR', 'occur_repo(s)'))
        for di in d_items[:20]:
            writer.writerow((di[0], round(di[1], 3), len(set(occur_repos[di[0]]))))
            print(f'{di[0]} & {round(di[1], 3)} & {len(set(occur_repos[di[0]]))} \\\\')


def _dump_sql_to_csv(repo_name):
    print(f'Merging {repo_name}...')
    doctype_path = os.path.join(doc_sql_dir, '%s.sqlite3' % (repo_name,))
    op_doc = SqliteOperator(doctype_path)
    doctype = dict()
    result = op_doc.select('select distinct * from data')
    for r in result:
        sql = 'select * from data where module=\'%s\' and qualname=\'%s\'' % (r[0], r[1])
        res = op_doc.select(sql)[0]
        doctype.setdefault(res[0], dict())[res[1]] = res[2].strip()
    op_doc.close()

    trace_path = os.path.join(trace_dir, '%s.sqlite3' % (repo_name,))
    op_ttype = SqliteOperator(trace_path)
    tmp_ttype = dict()
    stt = time.clock()
    for module, module_items in doctype.items():
        for qualname, qualname_items in module_items.items():
            sql = 'select arg_types from monkeytype_call_traces ' \
                  'where module=\'%s\' and qualname=\'%s\'' % (module, qualname)
            result = op_ttype.select(sql)
            if not result:
                continue
            for r in result:
                j = json.loads(r[0])
                for var_name in j:
                    tmp_ttype.setdefault(module, dict()).setdefault(qualname, dict()) \
                        .setdefault(var_name, set()).add(j[var_name]['qualname'])
    edt = time.clock()
    print(f'Merge {repo_name} cost {round(edt - stt, 3)} seconds.')

    op_ttype.close()

    ttype = dict()
    for module, module_items in tmp_ttype.items():
        for qualname, qualname_items in module_items.items():
            lines = []
            for var_name, ts in qualname_items.items():
                lines.append('%s  %s' % (var_name, ', '.join(ts)))
            lines = '\n'.join(lines)
            ttype.setdefault(module, dict())[qualname] = lines

    if not os.path.exists(doc_arc_dir):
        os.mkdir(doc_arc_dir)

    with open(os.path.join(doc_arc_dir, '%s.csv' % (repo_name,)), 'w', newline='', encoding='utf-8') as wf:
        writer = csv.writer(wf)
        writer.writerow(('module', 'qualname', 'ttype', 'docstring'))
        for module, module_items in ttype.items():
            for qualname, qualname_items in module_items.items():
                if module in doctype and qualname in doctype[module]:
                    writer.writerow((module, qualname, qualname_items, doctype[module][qualname].lower()))


def dump_sql_to_csv():
    if not os.path.exists(doc_sql_dir):
        os.mkdir(doc_sql_dir)
    for repo_name in repos:
        if os.path.exists(os.path.join(doc_arc_dir, '%s.csv' % (repo_name,))):
            continue
        _dump_sql_to_csv(repo_name)
