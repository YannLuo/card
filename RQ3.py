from RQ3 import card, rq3, baselines
import os


RQ3_result_dir = 'RQ3_result'


def main():
    if not os.path.exists(RQ3_result_dir):
        os.mkdir(RQ3_result_dir)
    vectorizer = card.train_card()
    rq3.start(vectorizer)
    baselines.compare('rf')
    baselines.compare('knn')


if __name__ == '__main__':
    main()
