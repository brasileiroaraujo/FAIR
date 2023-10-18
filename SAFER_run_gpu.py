import sys

import pandas as pd
import time
from streaming.controller_fairER_streaming import match_rank_streaming
import matcher



# ct=Consumer({'bootstrap.servers':'localhost:9092','group.id':'python-consumer','auto.offset.reset':'earliest'})
#
# print('Available topics to consume: ', ct.list_topics().topics)
#
# ct.subscribe(['target'])
################

def sortBySimilarity(element):
    return float(element[2])

def merge_clusters(clusters, incremental_clusters, k_ranking, ranking_mode):
    incremental_clusters.extend(clusters)
    incremental_clusters.sort(key=sortBySimilarity, reverse=True)
    if ranking_mode == 'm-fair':
        print("Running m-fair")
        return run_steraming_ranking_by_groups(incremental_clusters, 1, k_ranking) #1 = the first protected group
    else: #default fair-er ranking
        print("Running fair-er")
        return fairness_ranking(incremental_clusters, True, k_ranking) #True = first the protected group

def add_protected_group(dict_groups, index, tuple):
    if dict_groups.get(index) == None:
        dict_groups[index]=[tuple]
    else:
        dict_groups.get(index).append(tuple)

def run_steraming_ranking_by_groups(candidates, nextGroup, results_limit):
    matched_ids_left = set()
    matched_ids_right = set()
    matches = []

    # nonprotected_candidates = []
    grouped_candidates = {}
    for x in candidates:
        add_protected_group(grouped_candidates, int(x[3]), x) #index 0 is the non-protected group, others are protected

    #groups_indexes works as a interface to determine the netx group to be benefit (selected)
    #it works together nextGroup, which iterate over this list using the index (starting in 0)
    groups_indexes = list(range(0, max(grouped_candidates.keys())+1))

    while (groups_indexes) and (len(matches) < results_limit): #if groups_indexes is empty, means all groups are completely used
        if grouped_candidates.get(groups_indexes[nextGroup]) == None or len(grouped_candidates.get(groups_indexes[nextGroup])) == 0:
            groups_indexes.pop(nextGroup)
            nextGroup = 0 if nextGroup >= len(groups_indexes) else nextGroup
            continue

        cand = grouped_candidates.get(groups_indexes[nextGroup]).pop(0)

        # unique mapping constraint check
        if cand[0] in matched_ids_left or cand[1] in matched_ids_right:
            #print('Skipping candidate: ', cand, 'for violating unique mapping constraint')
            continue

        # add pair to matches
        matches.append(cand)
        matched_ids_left.add(cand[0])
        matched_ids_right.add(cand[1])

        if groups_indexes:#(nextGroup and nonprotected_candidates) or (not nextGroup and protected_candidates):
            nextGroup = 0 if nextGroup == (len(groups_indexes)-1) else nextGroup+1 # swap queues
            #print('swapping to ', 'protected' if nextProtected else 'nonprotected', 'queue')

    return matches

def fairness_ranking(candidates, nextProtected, results_limit):
    matched_ids_left = set()
    matched_ids_right = set()
    matches = []

    protected_candidates = [x for x in candidates if x[3]]
    nonprotected_candidates = [x for x in candidates if not x[3]]

    #print(protected_candidates)
    #print(nonprotected_candidates)

    while (protected_candidates or nonprotected_candidates) and (len(matches) < results_limit):
        cand = protected_candidates.pop(0) if ((nextProtected and protected_candidates) or not nonprotected_candidates) else nonprotected_candidates.pop(0)
        #print(cand)

        # unique mapping constraint check
        if cand[0] in matched_ids_left or cand[1] in matched_ids_right:
            #print('Skipping candidate: ', cand, 'for violating unique mapping constraint')
            continue

        # add pair to matches
        matches.append(cand)
        matched_ids_left.add(cand[0])
        matched_ids_right.add(cand[1])

        if (nextProtected and nonprotected_candidates) or (not nextProtected and protected_candidates):
            nextProtected = not nextProtected  # swap queues
            #print('swapping to ', 'protected' if nextProtected else 'nonprotected', 'queue')

    return matches


def setUpDitto(task, lm="distilbert", use_gpu=True, fp16="store_true",
               checkpoint_path='checkpoints/', dk=None, summarize="store_true", max_len=256, threshold=0):
    print("---- Configuring Ditto ----")
    # load the models
    matcher.set_seed(123)
    config, model = matcher.load_model(task, checkpoint_path,
                                       lm, use_gpu, fp16)

    summarizer = dk_injector = None
    if summarize:
        summarizer = matcher.Summarizer(config, lm)

    if dk is not None:
        if 'product' in dk:
            dk_injector = matcher.ProductDKInjector(config, dk)
        else:
            dk_injector = matcher.GeneralDKInjector(config, dk)

    # tune threshold
    if (threshold <= 0):
        print("---- Tuning Threshold ----")
        threshold = matcher.tune_threshold_parameterized(config, model, task, summarize, lm, max_len, dk)
    print('APPLIED THRESHOLD: ' + str(threshold))

    print("---- Ditto Configured ----")

    return config, model, threshold, summarizer, dk_injector

def open_csv(path):
    return pd.read_csv(path, header=None, sep='\n')

def main(args):
    list_of_pairs = []
    nextProtected = True
    BASE_PATH = args[0] #"D:/IntelliJ_Workspace/fairER/data/er_magellan/Structured/"
    k_ranking = int(args[1]) #20
    task = args[2] #'DBLP-GoogleScholar'
    lm= args[3] #"roberta"
    # k_batch = int(args[5]) #5742
    threshold = float(args[4])
    ranking_mode = args[7]
    incremental_clusters = []

    pairs_to_compare = open_csv(BASE_PATH + task + '/test.txt')

    config, model, threshold, summarizer, dk_injector = setUpDitto(task="Structured/" + task, lm=lm, checkpoint_path="checkpoints/", threshold = threshold)

    print("---- Ready to Consume ----")

    init = 0
    n_batches = int(args[5]) #5741
    window = n_batches - init
    pair_size = len(pairs_to_compare.index) - 1

    list_of_pairs = []

    while ((not pairs_to_compare.empty) and (init < pair_size)):#for i in range(0,len(preds.index)):
        lines = pairs_to_compare.loc[init:min(n_batches, pair_size)]

        for row in lines.to_numpy():
            triple = row[0].split("\t")
            data = [triple[0], triple[1]]#.loc[i].to_json()
            list_of_pairs.append(data)

        init = n_batches + 1
        n_batches = init + (window) #if (n_batches + (window) < len(source.index)) else dataset_size
        print("pairs size: " + str(len(lines.index)))
        # n_batches = n_batches + (window)

        #call ditto to match
        #PERFORM DITTO
        clusters, preds, av_time, nextProtected = match_rank_streaming(task, list_of_pairs, nextProtected, config, model, threshold, summarizer, dk_injector, lm, k_ranking, ranking_mode)

        #RANKING
        incremental_clusters = merge_clusters(clusters, incremental_clusters, k_ranking, ranking_mode)
        list_of_pairs = []
        # current_dataframe_source.drop(current_dataframe_source.index, inplace=True)

        print('clusters:')
        print(incremental_clusters)

        time.sleep(int(args[6]))

if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    main(args)