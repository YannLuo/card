from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from RQ3.corpus import ext_structed_docs, ext_unstructed_docs_with_one_arg
from utils.metric import jaccard


repos = ['asphalt', 'bs4', 'faker', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh']
basic_types = ['Dict', 'str', 'int', 'bool', 'List', 'bytes', 'Tuple', 'Callable', 'Type', 'NoneType'][:-1]
hint_terms = {
    'str': ['str', 'string', 'request', 'unicode', 'token', 'uri', 'code', 'name', 'method', 'url', 'identifier', 'body', 'text', 'content'],
    'int': ['int', 'integer', 'number', 'float', 'length'],
    'bool': ['True', 'true', 'TRUE', 'False', 'false', 'FALSE', 'bool', 'boolean', 'flag'],
    'Dict': ['dict', 'Dict', 'dictionary', 'Dictionary'],
    'List': ['List', 'list'],
    'bytes': ['file-like', 'sequence', 'byte', 'bytes'],
    'Tuple': ['tuple', 'Tuple'],
    'Callable': ['function', 'callable', 'Callable', 'call', 'called', 'helper'],
    'Type': ['type', 'entry', 'register', 'module'],
    # 'NoneType': ['byte', 'token', 'string', 'key', 'method'],
}
type_to_regexp = {
    'str': re.compile('\w+(\_)?name|\w+(\_)?method', re.I),
    'others1': re.compile('[a-z][a-z\_0-9]+\.[A-Z]\w{3,}'),
    'others2': re.compile('[A-Z]\w{5,}')
}


def train_card():
    docstrings = []
    for repo in repos:
        result = ext_structed_docs(repo) + ext_unstructed_docs_with_one_arg(repo)
        docstrings.extend([r[-1] for r in result])
    vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
    vectorizer.fit_transform(docstrings)
    return vectorizer


def card(corpus, vectorizer):
    jac_list = []
    id2term = {}
    for k, v in vectorizer.vocabulary_.items():
        id2term[v] = k
    for idx, item in enumerate(corpus):
        docstring = vectorizer.transform([item[-1]])[0]
        true_types = item[3]
        if set(true_types) & set(basic_types):
            # true_types at least have one basic type
            terms = [id2term[i] for i in docstring.indices]
            pred_types = []
            # try hint terms of each basic type whether match a type
            # term pattern `\w+(\_)?name|\w+(\_)?method`
            # is more likely to be a `str` type
            for basic_type in basic_types:
                h_terms = hint_terms[basic_type]
                if set(terms) & set(h_terms):
                    pred_types.append('List' if basic_type == 'Tuple' else basic_type)
                elif basic_type == 'str':
                    for term in terms:
                        if type_to_regexp[basic_type].match(term):
                            pred_types.append(basic_type)
                            break
            # if type `Dict` and `str` in pred_types at same time
            # remove `str` as type hardly never occur `Dict` and `str` at same time
            if 'Dict' in pred_types and 'str' in pred_types:
                pred_types.remove('str')
            # type `Type` often occur independently
            if 'Type' in pred_types and len(pred_types) > 1:
                pred_types.remove('Type')
            # if pred_types is empty
            # then find whether exists a term whose part of speech(pos) is NNS or NNPS
            # if found, the pos of this variable is likely to be `List`
            if not pred_types:
                pos_tags = [tag for w, tag in nltk.pos_tag(terms)]
                if set(['NNS', 'NNPS']) & set(pos_tags):
                    pred_types = ['List']

            pred_types = set(pred_types)
            included_types = set(basic_types) & set(true_types)
            if 'Tuple' in included_types:
                included_types -= set(['Tuple'])
                included_types.add('List')

            # calculate Jaccard Coefficient
            tmp = jaccard(included_types, pred_types)
            jac_list.append(tmp)
    # print(f'Mean jaccard: {0.0 if len(jac_list) == 0 else sum(jac_list) / len(jac_list):.3f} '
    #       f'consistent count: {sum([jac >= 1.0 for jac in jac_list])} total count: {len(jac_list)}')
    return jac_list
