from utils.sql import SqliteOperator
import os
import ast


def dispatch(node):
    if not node:
        return None
    node_type = type(node)
    if node_type is ast.Name:
        return node.id
    elif node_type is ast.NameConstant:
        return str(node.value)
    elif node_type is ast.Ellipsis:
        return '...'
    elif node_type is ast.Subscript:
        return '%s[%s]' % (dispatch(node.value), dispatch(node.slice))
    elif node_type is ast.Index:
        return dispatch(node.value)
    elif node_type is ast.Tuple:
        ret = '('
        if node.elts == 1:
            (elt, ) = node.elts
            ret += dispatch(elt)
            ret += ','
        else:
            # print([type(e) for e in node.elts])
            ret += ','.join([dispatch(e) for e in node.elts])
        ret += ')'
        return ret
    elif node_type is ast.List:
        ret = '['
        if node.elts == 1:
            (elt,) = node.elts
            ret += dispatch(elt)
            ret += ','
        else:
            ret += ','.join([dispatch(e) for e in node.elts])
        ret += ']'
        return ret


class ClassVisitor(ast.NodeVisitor):

    def __init__(self, file):
        self.file = file
        self.func_docs = []

    def visit_ClassDef(self, node):
        cls_name = node.name
        tmp_func_docs = extract_func_docstring(self.file, node, cls_name)
        self.func_docs.extend(tmp_func_docs)


def extract_func_docstring(file, node, cls_name=None):
    func_docs = []
    for b in node.body:
        if isinstance(b, ast.FunctionDef):
            func_name = b.name
            arguments = b.args
            if arguments:
                args = arguments.args
                kwonlyargs = arguments.kwonlyargs
                args_type = []
                kwonlyargs_type = []
                for arg in args:
                    args_type.append(dispatch(arg.annotation))
                for arg in kwonlyargs:
                    kwonlyargs_type.append(dispatch(arg.annotation))
            for bi in b.body:
                if isinstance(bi, ast.Expr) and isinstance(bi.value, ast.Str):
                    func_docs.append(
                        (file,
                         cls_name,
                         func_name,
                         bi.value.s,
                         list(zip([arg.arg for arg in args], args_type)),
                         list(zip([arg.arg for arg in kwonlyargs], kwonlyargs_type)))
                    )
    return func_docs


def extract_docstring_from_file(file):
    with open(file, 'r', encoding='utf-8') as rf:
        content = rf.read()
    root = ast.parse(content)
    result = extract_func_docstring(file, root)
    cls_visitor = ClassVisitor(file)
    cls_visitor.visit(root)
    result.extend(cls_visitor.func_docs)
    return result


def dfs(cur_dir):

    def dfs_dir(cur_dir):
        if not os.path.isdir(cur_dir):
            return
        files = os.listdir(cur_dir)
        for file in files:
            new_path = os.path.join(cur_dir, file)
            if os.path.isdir(new_path):
                dfs_dir(new_path)
            elif file.endswith('.py'):
                result.extend(extract_docstring_from_file(new_path))

    result = []
    dfs_dir(cur_dir)
    return result


def main():
    repos = ['asphalt', 'hbmqtt', 'httpie', 'oauthlib', 'pycookiecheat', 'pydantic', 'requests', 'sh', 'bs4', 'faker']
    doc_dir = 'doc_sql'
    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)
    for repo_name in repos:
        sql_file_path = os.path.join(doc_dir, f'{repo_name}.sqlite3')
        if os.path.exists(sql_file_path):
            os.remove(sql_file_path)
        op_sql = SqliteOperator(sql_file_path)
        op_sql.execute('create table data(module VARCHAR, qualname VARCHAR, doc_sql VARCHAR)', None)
        result = dfs(os.path.join('srcs', repo_name))
        for r in result:
            file = r[0]
            cls = r[1]
            func_name = r[2]
            doc = r[3]
            path = file[:-3].split('\\')[1:]
            if path[0] == 'sh':
                path = path[1:]
            module = '.'.join(path)
            qualname = '%s%s' % (cls + '.' if cls else '', func_name)
            op_sql.execute('insert into data values(?, ?, ?)', (module, qualname, doc))
        op_sql.commit()
        op_sql.close()
        print(f'Docstrings of {repo_name} dumped.')


if __name__ == '__main__':
    main()
