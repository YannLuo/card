import os
import json
import csv
from utils.sql import SqliteOperator


repos = ['asphalt', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh', 'bs4', 'faker']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType']


def dump_true_types():
    true_type_dir = 'trace_sql'
    with open('tmp.csv', 'w', encoding='utf-8', newline='') as wf:
        writer = csv.writer(wf)
        for repo_name in repos:
            print(repo_name)
            ttype_path = os.path.join(true_type_dir, '%s.sqlite3' % (repo_name,))
            op_ttype = SqliteOperator(ttype_path)
            result = op_ttype.select('select module, qualname, arg_types from monkeytype_call_traces')
            d = {}
            for r in result:
                module = r[0]
                qualname = r[1]
                var_json = json.loads(r[2])
                for var_name in var_json:
                    mtype = var_json[var_name]['qualname']
                    if mtype in basic_types:
                        d.setdefault(f'{module} {qualname} {var_name}', set()).add(mtype)
            for k, v in d.items():
                tl = list(v)
                if len(tl) > 1:
                    writer.writerow(tl)


if __name__ == '__main__':
    dump_true_types()
