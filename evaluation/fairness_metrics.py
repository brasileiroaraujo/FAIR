import math
import numpy as np
import util
from evaluation import decompose_col_val
from evaluation.decompose_col_val import decompose_srt_to_df


def compute_bias(clusters, dataset):
    k = len(clusters)
    R = {}

    #start the groups
    for i in range(0, util.number_of_groups(dataset)):
        R[i] = 0
    for pair in clusters:
        R[pair[3]] = R[pair[3]] + 1
        # if (pair[3] in R.keys()):
        #     R[pair[3]] = R[pair[3]] + 1
        # else:
        #     R[pair[3]] = 1

    values = list(R.values())
    print(values)

    #compute the max difference between the groups
    max_unfairness = (max(values)/k) - (min(values)/k)

    return max_unfairness

def compute_tp_fn_by_group_top_k(clusters, labed_file, dataset, ranking_mode):
    tp = 0
    fn = 0
    tp_fn = {}
    max_recall_per_group = {}
    goldstandard = [line for line in labed_file if line[len(line)-1] == '1']

    print(len(clusters))
    #compute tps
    for pair in clusters:
        for result in labed_file:
            if(pair[0].strip() == result.split('\t')[0].split('VAL')[1].split('COL')[0].strip() and pair[1].strip() == result.split('\t')[1].split('VAL')[1].split('COL')[0].strip()):
                if(result[len(result)-1] == '1'):
                    if (pair[3] in tp_fn.keys()):
                        tp_fn[pair[3]][0] = tp_fn[pair[3]][0] + 1
                    else:
                        tp_fn[pair[3]] = [1, 0]
                    break #do not need to continue

    #compute the max recall in each group
    for pair in goldstandard:
        pair_df = decompose_srt_to_df(pair)
        group = util.pair_is_protected_by_group(pair_df, dataset, False) if ranking_mode == 'm-fair' or ranking_mode == 'none' \
            else util.pair_is_protected(pair_df, dataset, False)
        if (group in max_recall_per_group.keys()):
            max_recall_per_group[group] = max_recall_per_group[group] + 1
        else:
            max_recall_per_group[group] = 1

    discrepancy_between_groups = len(clusters)%len(tp_fn) # compute if the top-k will generate discrepancy between groups. e.g., top-10 with 4 groups will generate 2 groups with +1 pair each
    max_recall_top_k = math.trunc(len(clusters)/len(tp_fn)) #since there is top-k values, the max recall is divided by groups

    for key in tp_fn.keys():
        if key in range(1,discrepancy_between_groups+1): #this groups will benefit by the discrepancy, with one more pair
            fn = min(max_recall_top_k+1, max_recall_per_group[key]) - tp_fn[key][0]
        else:
            fn = min(max_recall_top_k, max_recall_per_group[key]) - tp_fn[key][0]

        if fn < 0: #normalize to avoid negative values for unbalanced groups
            fn = 0
        tp_fn[key][1] = fn

    print(tp_fn)
    return tp_fn

#In top-k approaches (i.e., the max recall possible for the top-k), we assume the true canditates (goldstandard) as the max between the size of the goldstandard and the k value
def compute_TPRP_top_k(clusters, labed_file, dataset, ranking_mode):
    k = len(clusters)
    tp_fn_per_group = compute_tp_fn_by_group_top_k(clusters, labed_file, dataset, ranking_mode)
    for key in tp_fn_per_group.keys():
        tp, fn = tp_fn_per_group.get(key)
        tpr = tp/(tp+fn)
        tp_fn_per_group[key] = tpr
    print(tp_fn_per_group)

    values = list(tp_fn_per_group.values())

    print(values)
    mean_value = np.mean(values)

    #compute the max difference between the groups
    tprp = 1.0 - mean_value

    return tprp


def compute_tp_fp_by_group(clusters, goldstandard):
    tp = 0
    fp = 0
    tp_fp = {}

    print(len(clusters))

    for pair in clusters:
        for result in goldstandard:
            if(pair[0].strip() == result.split('\t')[0].split('VAL')[1].split('COL')[0].strip() and pair[1].strip() == result.split('\t')[1].split('VAL')[1].split('COL')[0].strip()):
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
    mean_value = np.mean(values)

    #compute the max difference between the groups and the ideal value (i.e., 1)
    ppvp = 1.0 - mean_value

    return ppvp

# clusters = [('adobe premiere pro cs3', 'adobe premiere pro cs3', 0.9656693935394287, 0), ('adobe dreamweaver cs3 upgrade', 'adobe dreamweaver cs3 upgrade', 0.9655036926269531, 0), ('icopydvds2 ultra by digital wunders', 'global marketing partners icopydvds2 ultra by digital wunders', 0.9655009508132935, 0), ('adobe acrobat distiller svr v6-cd sun 100u 42050106 )', 'adobe 42050106 acrobat distiller svr v6-cd sun 100u', 0.9654878973960876, 0), ('acad corel painter x pc/mac', 'corel painter x', 0.965453565120697, 0), ('intervideo windvd 8 platinum', 'corel intervideo windvd 8 platinum software for windows authoring software', 0.9654138088226318, 0), ('adobe premiere pro cs3', 'adobe premiere pro cs3 video editing software for windows professional editing software', 0.9654093384742737, 0), ('reel deal slots mystic forest', 'phantom efx reel deal slots mystic forest', 0.9653741717338562, 0), ('emedia intermediate guitar method win/mac', 'emedia music corp emedia intermediate guitar method', 0.9653590321540833, 0), ('bias peak le 5', 'bias peak le 5 software music production software', 0.9653339385986328, 0), ('sony sound forge audio studio 8', 'sony media software sound forge audio studio 8 software music production software', 0.9653218388557434, 2), ('red hat enterprise linux ws v. 4 update 4 license 1 workstation rl296aa )', 'rl296aa red hat enterprise linux ws v. 4 update 4 license 1 workstation', 0.9652944803237915, 0), ('musicalis guitar workshop', 'musicalis universal guitar workshop', 0.9652790427207947, 0), ('rome total war gold edition', 'sega of america inc rome total war gold edition', 0.9652754068374634, 0), ('xtreme photostory on cd and dvd 6', 'magix entertainment corp. xtreme photostory on cd and dvd 6', 0.9652734994888306, 0), ('adobe after effects cs3', 'adobe after effects cs3 professional software for mac effects software', 0.9652574062347412, 0), ('sony super duper music looper', 'sony media software super duper music looper software music production software', 0.965206503868103, 2), ('genuine fractals genuine fractals print pro 1u', 'onone software genuine fractals print pro software', 0.9652037620544434, 0), ('adobe after effects cs3', 'adobe after effects cs3 professional software for windows effects software', 0.9652023911476135, 0), ('adobe premiere pro cs3', 'adobe premiere pro cs3 video editing software for mac av production software', 0.9651574492454529, 0)]
# with open('D:/IntelliJ_Workspace/fairER/data/er_magellan/Structured/Amazon-Google/test.txt', encoding="utf8") as file:
#     labed_file = [line.rstrip() for line in file]
#
# k = len(clusters)
# tp_fp_per_group = compute_tp_fp_by_group(clusters, labed_file)
# for key in tp_fp_per_group.keys():
#     tp, fp = tp_fp_per_group.get(key)
#     ppv = tp/(tp+fp)
#     tp_fp_per_group[key] = ppv
# print(tp_fp_per_group)
#
# values = list(tp_fp_per_group.values())
# mean_value = np.mean(values)
#
# #compute the max difference between the groups and the ideal value (i.e., 1)
# ppvp = 1.0 - mean_value
#
# print(ppvp)



# def compute_tp_fn_by_group(clusters, goldstandard, dataset, ranking_mode):
#     tp = 0
#     fn = 0
#     tp_fn = {}
#     visited = []
#
#     print(len(clusters))
#
#     for result in goldstandard:
#         for pair in clusters:
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

# def compute_TPRP(clusters, goldstandard, dataset, ranking_mode):
#     k = len(clusters)
#     tp_fn_per_group = compute_tp_fn_by_group(clusters, goldstandard, dataset, ranking_mode)
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