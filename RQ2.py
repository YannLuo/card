from RQ2 import archive_docstrings, dump_doc_sql, rq2
import os


RQ2_result_dir = 'RQ2_result'


def main():
    if not os.path.exists(RQ2_result_dir):
        os.mkdir(RQ2_result_dir)
    dump_doc_sql.main()
    archive_docstrings.dump_sql_to_csv()
    rq2.main()


if __name__ == '__main__':
    main()
