from matching import run_deepmatcher as dm
from matching import run_deepmatcher_w_mojito as dmm
import os
import util
import pandas as pd
from clustering import fair_unique_mapping_clustering as fumc
import sys
sys.path.append(sys.path.abspath('C:\\Users\\admin\\IntelliJ_Workspace'))
import matcher


import web.library.methods as methods


def open_ditto_result(path):
    df = pd.read_json(path_or_buf=path, lines=True)

    df = pd.concat([df,extractLeftColumns(df)], axis=1)
    df = df.drop(axis=1, columns=["left"])

    df = pd.concat([df,extractRightColumns(df)], axis=1)
    df = df.drop(axis=1, columns=["right"])

    df.rename(columns={"match_confidence": 'match_score', "match": 'label'}, inplace=True)
    return df

def extractLeftColumns(df):
    df_splited = df.left.str.split("COL", expand=True)
    df_splited = df_splited.drop(axis=1, columns=[0])
    df_splited.rename(columns={1: 'left_Beer_Name', 2: 'left_Brew_Factory_Name', 3: 'left_Style', 4: 'left_ABV'}, inplace=True)

    df_splited['left_Beer_Name'] = df_splited['left_Beer_Name'].str.split("VAL", expand=True)[1]
    df_splited['left_Brew_Factory_Name'] = df_splited['left_Brew_Factory_Name'].str.split("VAL", expand=True)[1]
    df_splited['left_Style'] = df_splited['left_Style'].str.split("VAL", expand=True)[1]
    df_splited['left_ABV'] = df_splited['left_ABV'].str.split("VAL", expand=True)[1]

    return df_splited

def extractRightColumns(df):
    df_splited = df.right.str.split("COL", expand=True)
    df_splited = df_splited.drop(axis=1, columns=[0])
    df_splited.rename(columns={1: 'right_Beer_Name', 2: 'right_Brew_Factory_Name', 3: 'right_Style', 4: 'right_ABV'}, inplace=True)

    df_splited['right_Beer_Name'] = df_splited['right_Beer_Name'].str.split("VAL", expand=True)[1]
    df_splited['right_Brew_Factory_Name'] = df_splited['right_Brew_Factory_Name'].str.split("VAL", expand=True)[1]
    df_splited['right_Style'] = df_splited['right_Style'].str.split("VAL", expand=True)[1]
    df_splited['right_ABV'] = df_splited['right_ABV'].str.split("VAL", expand=True)[1]

    return df_splited


def run(data, data_path, train_file, valid_file, test_file, explanation, k_results):
    ###########
    # Matching
    ###########
    # if not os.path.exists(data_path + '/dm_results.csv'):
    #     if explanation:
    #         preds = dmm.run(data_path, train_file, valid_file, test_file)
    #     else:
    #         preds = dm.run(data_path, train_file, valid_file,
    #                        test_file)  # , unlabeled_file)
    #     preds.to_csv(data_path + '/dm_results.csv')

    # unnecessary read for the 1st time, but throws error otherwise
    preds = open_ditto_result(data_path + '\output_small_test2.jsonl')#Ditto
    # preds = pd.read_csv(data_path + '/dm_results.csv')#FairER
    # print("Initial Ranking:\n", preds[:k_results].to_string(index=False))

    # Ranking of matching results in desc. match score
    preds = preds.sort_values(by='match_score', ascending=False)
    # print("Desc. Ranking:\n", preds[:k_results].to_string(index=False))

    ################################
    # Fair Unique Mapping Clustering
    ################################
    # initial_pairs = [(int(a.id.split('_')[0]), int(a.id.split('_')[1]), a.match_score, util.pair_is_protected(a, data, False, explanation))
    #                  for a in preds.itertuples(index=False)] #FairEr

    initial_pairs = [(a.left_Beer_Name, a.right_Beer_Name, a.match_score, util.pair_is_protected(a, data, False, explanation))
                     for a in preds.itertuples(index=False)] #Ditto

    clusters = fumc.run(initial_pairs, k_results)
    print("\nclustering results:\n", clusters)


    # # Write clusters to json file
    # methods.clusters_to_json(clusters)
    # # Write preds to json file
    # methods.preds_to_json(data_path)

    return clusters, preds




def run_streaming(data, data_frame, nextProtected, k_results):
    ###########
    # Matching
    ###########
    # if not os.path.exists(data_path + '/dm_results.csv'):
    #     if explanation:
    #         preds = dmm.run(data_path, train_file, valid_file, test_file)
    #     else:
    #         preds = dm.run(data_path, train_file, valid_file,
    #                        test_file)  # , unlabeled_file)
    #     preds.to_csv(data_path + '/dm_results.csv')

    # Ranking of matching results in desc. match score
    preds = data_frame.sort_values(by='match_score', ascending=False)
    # print("Desc. Ranking:\n", preds[:k_results].to_string(index=False))

    ################################
    # Fair Unique Mapping Clustering
    ################################
    # initial_pairs = [(int(a.id.split('_')[0]), int(a.id.split('_')[1]), a.match_score, util.pair_is_protected(a, data, False, explanation))
    #                  for a in preds.itertuples(index=False)] #FairEr

    initial_pairs = [(a.left_Beer_Name, a.right_Beer_Name, a.match_score, util.pair_is_protected(a, data, False))
                     for a in preds.itertuples(index=False)] #Ditto

    clusters, nextProtected = fumc.run_steraming(initial_pairs, nextProtected, k_results)
    print("\nclustering results:\n", clusters)


    # # Write clusters to json file
    # methods.clusters_to_json(clusters)
    # # Write preds to json file
    # methods.preds_to_json(data_path)

    return clusters, preds, nextProtected


def run_matching_ranking_streaming(data, data_frame, nextProtected, k_results):
    ###########
    # Matching with Ditto
    ###########
    print("--- Matching with Ditto ...  ---")
    matcher.run_matcher(task="Structured/Beer", input_path="data/er_magellan/Structured/Beer/test.txt", output_path="output/output_small_test_fair.jsonl", lm="roberta", checkpoint_path="checkpoints/")







    # # Ranking of matching results in desc. match score
    # preds = data_frame.sort_values(by='match_score', ascending=False)
    # # print("Desc. Ranking:\n", preds[:k_results].to_string(index=False))
    #
    # ################################
    # # Fair Unique Mapping Clustering
    # ################################
    # # initial_pairs = [(int(a.id.split('_')[0]), int(a.id.split('_')[1]), a.match_score, util.pair_is_protected(a, data, False, explanation))
    # #                  for a in preds.itertuples(index=False)] #FairEr
    #
    # initial_pairs = [(a.left_Beer_Name, a.right_Beer_Name, a.match_score, util.pair_is_protected(a, data, False))
    #                  for a in preds.itertuples(index=False)] #Ditto
    #
    # clusters, nextProtected = fumc.run_steraming(initial_pairs, nextProtected, k_results)
    # print("\nclustering results:\n", clusters)
    #
    #
    # # # Write clusters to json file
    # # methods.clusters_to_json(clusters)
    # # # Write preds to json file
    # # methods.preds_to_json(data_path)

    # return clusters, preds, nextProtected