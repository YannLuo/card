from RQ3 import rq3, baselines
import os


RQ3_result_dir = 'RQ3_result'


def main():
    if not os.path.exists(RQ3_result_dir):
        os.mkdir(RQ3_result_dir)
    rq3.start()
    baselines.baseline('rf')
    baselines.baseline('knn')


if __name__ == '__main__':
    main()
