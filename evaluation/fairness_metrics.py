import util
from evaluation import decompose_col_val


def compute_bias(clusters):
    k = len(clusters)
    R = {}
    for pair in clusters:
        if (pair[3] in R.keys()):
            R[pair[3]] = R[pair[3]] + 1
        else:
            R[pair[3]] = 1

    values = list(R.values())
    print(values)

    #compute the max difference between the groups
    max_unfairness = (max(values)/k) - (min(values)/k)

    return max_unfairness

# def compute_tp_fn_by_group_top_k(clusters, goldstandard, dataset, ranking_mode):
#     tp = 0
#     fn = 0
#     tp_fn = {}
#     ordered_goldstandard = []
#
#     print(len(clusters))
#     for pair in clusters:
#         for result in goldstandard:
#             if pair[0] == result.split('\t')[0].split('VAL')[1].split('COL')[0] and pair[1] == result.split('\t')[1].split('VAL')[1].split('COL')[0] and pair not in visited:
#                 if result[len(result) - 1] == '1':
#                     if (pair[3] in tp_fn.keys()):
#                         tp_fn[pair[3]][0] = tp_fn[pair[3]][0] + 1
#                     else:
#                         tp_fn[pair[3]] = [1, 0]
#                     visited.append(pair)
#                     break #do not need to continue
#         #the true condition was not found
#         if(result[len(result)-1] == '1'):
#             group = util.pair_is_protected_by_group(decompose_col_val.decompose_srt_to_df(result), dataset, False) \
#                 if ranking_mode == 'm-fair' else util.pair_is_protected(decompose_col_val.decompose_srt_to_df(result), dataset, False)#util.pair_is_protected_by_group(result, dataset, False)
#             if (group in tp_fn.keys()):
#                 tp_fn[group][1] = tp_fn[group][1] + 1
#             else:
#                 tp_fn[pair[3]] = [0, 1]
#
#     print(tp_fn)
#     return tp_fn
#
# #In top-k approaches (i.e., the max recall possible for the top-k), we assume the true canditates (goldstandard) as the max between the size of the goldstandard and the k value
# def compute_TPRP_top_k(clusters, goldstandard, dataset, ranking_mode):
#     k = len(clusters)
#     tp_fn_per_group = compute_tp_fn_by_group_top_k(clusters, goldstandard, dataset, ranking_mode)
#     for key in tp_fn_per_group.keys():
#         tp, fn = tp_fn_per_group.get(key)
#         tpr = tp/(tp+fn)
#         tp_fn_per_group[key] = tpr
#     print(tp_fn_per_group)
#
#     values = list(tp_fn_per_group.values())
#
#     print(values)
#
#     #compute the max difference between the groups
#     tprp = max(values) - min(values)
#
#     return tprp

def compute_tp_fn_by_group(clusters, goldstandard, dataset, ranking_mode):
    tp = 0
    fn = 0
    tp_fn = {}
    visited = []

    print(len(clusters))

    for result in goldstandard:
        for pair in clusters:
            if pair[0] == result.split('\t')[0].split('VAL')[1].split('COL')[0] and pair[1] == result.split('\t')[1].split('VAL')[1].split('COL')[0] and pair not in visited:
                if result[len(result) - 1] == '1':
                    if (pair[3] in tp_fn.keys()):
                        tp_fn[pair[3]][0] = tp_fn[pair[3]][0] + 1
                    else:
                        tp_fn[pair[3]] = [1, 0]
                    visited.append(pair)
                    break #do not need to continue
        #the true condition was not found
        if(result[len(result)-1] == '1'):
            group = util.pair_is_protected_by_group(decompose_col_val.decompose_srt_to_df(result), dataset, False) \
                if ranking_mode == 'm-fair' else util.pair_is_protected(decompose_col_val.decompose_srt_to_df(result), dataset, False)#util.pair_is_protected_by_group(result, dataset, False)
            if (group in tp_fn.keys()):
                tp_fn[group][1] = tp_fn[group][1] + 1
            else:
                tp_fn[pair[3]] = [0, 1]

    print(tp_fn)
    return tp_fn

def compute_TPRP(clusters, goldstandard, dataset, ranking_mode):
    k = len(clusters)
    tp_fn_per_group = compute_tp_fn_by_group(clusters, goldstandard, dataset, ranking_mode)
    for key in tp_fn_per_group.keys():
        tp, fn = tp_fn_per_group.get(key)
        tpr = tp/(tp+fn)
        tp_fn_per_group[key] = tpr
    print(tp_fn_per_group)

    values = list(tp_fn_per_group.values())

    print(values)

    #compute the max difference between the groups
    tprp = max(values) - min(values)

    return tprp


def compute_tp_fp_by_group(clusters, goldstandard):
    tp = 0
    fp = 0
    tp_fp = {}

    print(len(clusters))

    for pair in clusters:
        for result in goldstandard:
            if(pair[0] == result.split('\t')[0].split('VAL')[1].split('COL')[0] and pair[1] == result.split('\t')[1].split('VAL')[1].split('COL')[0]):
                if(result[len(result)-1] == '1'):
                    if (pair[3] in tp_fp.keys()):
                        tp_fp[pair[3]][0] = tp_fp[pair[3]][0] + 1
                    else:
                        tp_fp[pair[3]] = [1, 0]
                    break #do not need to continue
                else:
                    if (pair[3] in tp_fp.keys()):
                        tp_fp[pair[3]][1] = tp_fp[pair[3]][1] + 1
                    else:
                        tp_fp[pair[3]] = [0, 1]
                    break #do not need to continue
                # print(result)

    print(tp_fp)
    return tp_fp


def compute_PPVP(clusters, goldstandard):
    k = len(clusters)
    tp_fp_per_group = compute_tp_fp_by_group(clusters, goldstandard)
    for key in tp_fp_per_group.keys():
        tp, fp = tp_fp_per_group.get(key)
        ppv = tp/(tp+fp)
        tp_fp_per_group[key] = ppv
    print(tp_fp_per_group)

    values = list(tp_fp_per_group.values())

    #compute the max difference between the groups
    ppvp = max(values) - min(values)

    return ppvp