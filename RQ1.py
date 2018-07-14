from RQ1 import freq_types, rq1
import os
from subprocess import Popen


RQ1_result_dir = 'RQ1_result'


def main():
    if not os.path.exists(RQ1_result_dir):
        os.mkdir(RQ1_result_dir)
    freq_types.main()
    rq1.dump_true_types()
    with open(os.path.join(RQ1_result_dir, 'association_rules.txt'), mode='w', encoding='utf-8') as wf:
        p = Popen('python RQ1/apriori.py -f tmp.csv -s 0.1 -c 0.5', stdout=wf)
        p.wait()
    os.remove('tmp.csv')


if __name__ == '__main__':
    main()
