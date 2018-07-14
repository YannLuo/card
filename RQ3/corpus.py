import re
from nltk import sent_tokenize
import csv


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
RQ3_result_dir = 'RQ3_result'


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


