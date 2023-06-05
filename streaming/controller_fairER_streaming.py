import os
import time
import sys

import evaluation.accuracy as eval
import evaluation.fairness as f_eval
from streaming.fairER_streaming import run, run_streaming, run_matching_ranking_streaming

import web.library.methods as methods

BASE_PATH = "D:\\IntelliJ_Workspace\\fairER\\"

# This pipeline performs a Fair version (for streaming data) of Unique Mapping Clustering (creates two PQs instead of one)
# instead of running the fa*ir algorithm for re-ranking
def match(data_path, train_file, valid_file, test_file, explanation):
    k = 20
    data = os.path.basename(data_path)

    print('\n', data, '\n')

    av_time = 0
    for _ in range(10):
        start_time = time.time()
        clusters, preds = run(data, data_path, train_file,
                              valid_file, test_file, int(explanation), k)
        ex_time = time.time() - start_time
        av_time += ex_time

    #############################
    # Evaluation
    #############################
    print("--- %s seconds ---" % (av_time / 10.0))

    accuracy = eval.get_accuracy(clusters, preds)
    print("accuracy:", accuracy)

    # spd = f_eval.get_spd(clusters, preds, data)
    # print("SPD:", spd)
    #
    # eod = f_eval.get_eod(clusters, preds, data)
    # print("EOD:", eod)
    print()


    # # Write evaluation results to json file
    # methods.eval_to_json(accuracy, spd, eod)


def match_streaming(data, data_frame, nextProtected):
    k = 20
    print('\n', 'Streaming processing ... ' + data, '\n')
    # data_path = os.path.join(BASE_PATH, 'resources','Datasets',data)

    av_time = 0
    # for _ in range(10):
    start_time = time.time()
    clusters, preds, nextProtected = run_streaming(data, data_frame, nextProtected, k)
    ex_time = time.time() - start_time
    av_time += ex_time

    return clusters, preds, av_time, nextProtected

def match_rank_streaming(data, list_of_pairs, nextProtected):
    k = 20
    print('\n', 'Streaming processing ... ' + data, '\n')
    # data_path = os.path.join(BASE_PATH, 'resources','Datasets',data)

    av_time = 0
    # for _ in range(10):
    start_time = time.time()
    clusters, preds, nextProtected = run_matching_ranking_streaming(data, list_of_pairs, nextProtected, k)
    ex_time = time.time() - start_time
    av_time += ex_time

    return clusters, preds, av_time, nextProtected


if __name__ == '__main__':
    args = len(sys.argv) > 5

    datasets_path = sys.argv[1] if args else os.path.join(BASE_PATH, 'resources','Datasets','Beer')

    train_file = sys.argv[2] if args else 'joined_train.csv'
    valid_file = sys.argv[3] if args else 'joined_valid.csv'
    test_file = sys.argv[4] if args else 'joined_test.csv'
    explanation = sys.argv[5] if args else 0

    match(datasets_path, train_file, valid_file, test_file, explanation)
