import time

# import data_distribution_assessment.data_assessment_menager
import data_sender
from decompose_col_val import decompose_srt_to_full_df, format_gnem_output_to_df
# from matching import run_deepmatcher as dm
# from matching import run_deepmatcher_w_mojito as dmm
import os
import util
import pandas as pd
from clustering import fair_unique_mapping_clustering as fumc
import sys
sys.path.insert(1, '../ditto')
import matcher
import torch
import torch.nn.functional as F


# import web.library.methods as methods


def open_ditto_result(path, data):
    df = pd.read_json(path_or_buf=path, lines=True)

    if df.empty:
        return df

    df = pd.concat([df,extractLeftColumns(df, data)], axis=1)
    df = df.drop(axis=1, columns=["left"])

    df = pd.concat([df,extractRightColumns(df, data)], axis=1)
    df = df.drop(axis=1, columns=["right"])

    df.rename(columns={"match_confidence": 'match_score', "match": 'label'}, inplace=True)
    return df

def extractLeftColumns(df, data):
    df_splited = df.left.str.split("COL", expand=True)
    df_splited = df_splited.drop(axis=1, columns=[0])

    if data == 'Beer':
        df_splited.rename(columns={1: 'left_Beer_Name', 2: 'left_Brew_Factory_Name', 3: 'left_Style', 4: 'left_ABV'}, inplace=True)
        df_splited['left_Beer_Name'] = df_splited['left_Beer_Name'].str.split("VAL", expand=True)[1]
        df_splited['left_Brew_Factory_Name'] = df_splited['left_Brew_Factory_Name'].str.split("VAL", expand=True)[1]
        df_splited['left_Style'] = df_splited['left_Style'].str.split("VAL", expand=True)[1]
        df_splited['left_ABV'] = df_splited['left_ABV'].str.split("VAL", expand=True)[1]
    elif data == 'Amazon-Google':
        df_splited.rename(columns={1: 'left_title', 2: 'left_manufacturer', 3: 'left_price'}, inplace=True)
        df_splited['left_title'] = df_splited['left_title'].str.split("VAL", expand=True)[1]
        df_splited['left_manufacturer'] = df_splited['left_manufacturer'].str.split("VAL", expand=True)[1]
        df_splited['left_price'] = df_splited['left_price'].str.split("VAL", expand=True)[1]
    elif data == 'DBLP-ACM':
        df_splited.rename(columns={1: 'left_title', 2: 'left_authors', 3: 'left_venue', 4: 'left_year'}, inplace=True)
        df_splited['left_title'] = df_splited['left_title'].str.split("VAL", expand=True)[1]
        df_splited['left_authors'] = df_splited['left_authors'].str.split("VAL", expand=True)[1]
        df_splited['left_venue'] = df_splited['left_venue'].str.split("VAL", expand=True)[1]
        df_splited['left_year'] = df_splited['left_year'].str.split("VAL", expand=True)[1]
    elif data == 'DBLP-GoogleScholar':
        df_splited.rename(columns={1: 'left_title', 2: 'left_authors', 3: 'left_venue', 4: 'left_year'}, inplace=True)
        df_splited['left_title'] = df_splited['left_title'].str.split("VAL", expand=True)[1]
        df_splited['left_authors'] = df_splited['left_authors'].str.split("VAL", expand=True)[1]
        df_splited['left_venue'] = df_splited['left_venue'].str.split("VAL", expand=True)[1]
        df_splited['left_year'] = df_splited['left_year'].str.split("VAL", expand=True)[1]
    elif data == 'iTunes-Amazon':
        df_splited.rename(columns={1: 'left_Song_Name', 2: 'left_Artist_Name', 3: 'left_Album_Name', 4: 'left_Genre', 5: 'left_Price', 6:'left_Copyleft', 7:'left_Time', 8:'left_Released'}, inplace=True)
        df_splited['left_Song_Name'] = df_splited['left_Song_Name'].str.split("VAL", expand=True)[1]
        df_splited['left_Artist_Name'] = df_splited['left_Artist_Name'].str.split("VAL", expand=True)[1]
        df_splited['left_Album_Name'] = df_splited['left_Album_Name'].str.split("VAL", expand=True)[1]
        df_splited['left_Genre'] = df_splited['left_Genre'].str.split("VAL", expand=True)[1]
        df_splited['left_Price'] = df_splited['left_Price'].str.split("VAL", expand=True)[1]
        df_splited['left_Copyleft'] = df_splited['left_Copyleft'].str.split("VAL", expand=True)[1]
        df_splited['left_Time'] = df_splited['left_Time'].str.split("VAL", expand=True)[1]
        df_splited['left_Released'] = df_splited['left_Released'].str.split("VAL", expand=True)[1]
    elif data == 'Walmart-Amazon':
        df_splited.rename(columns={1: 'left_title', 2: 'left_category', 3: 'left_brand', 4: 'left_modelno', 5: 'left_price'}, inplace=True)
        df_splited['left_title'] = df_splited['left_title'].str.split("VAL", expand=True)[1]
        df_splited['left_category'] = df_splited['left_category'].str.split("VAL", expand=True)[1]
        df_splited['left_brand'] = df_splited['left_brand'].str.split("VAL", expand=True)[1]
        df_splited['left_modelno'] = df_splited['left_modelno'].str.split("VAL", expand=True)[1]
        df_splited['left_price'] = df_splited['left_price'].str.split("VAL", expand=True)[1]
    else:
        print("NEED TO IMPLEMENT CONDITION")

    # for i in range(1, len(df_splited.columns) + 1):
    #     df_splited['l_' + str(i)] = df_splited[i].str.split("VAL", expand=True)[1]
    #     df_splited = df_splited.drop(axis=1, columns=[i])


    return df_splited

def extractRightColumns(df, data):
    df_splited = df.right.str.split("COL", expand=True)
    df_splited = df_splited.drop(axis=1, columns=[0])

    if data == 'Beer':
        df_splited.rename(columns={1: 'right_Beer_Name', 2: 'right_Brew_Factory_Name', 3: 'right_Style', 4: 'right_ABV'}, inplace=True)
        df_splited['right_Beer_Name'] = df_splited['right_Beer_Name'].str.split("VAL", expand=True)[1]
        df_splited['right_Brew_Factory_Name'] = df_splited['right_Brew_Factory_Name'].str.split("VAL", expand=True)[1]
        df_splited['right_Style'] = df_splited['right_Style'].str.split("VAL", expand=True)[1]
        df_splited['right_ABV'] = df_splited['right_ABV'].str.split("VAL", expand=True)[1]
    elif data == 'Amazon-Google':
        df_splited.rename(columns={1: 'right_title', 2: 'right_manufacturer', 3: 'right_price'}, inplace=True)
        df_splited['right_title'] = df_splited['right_title'].str.split("VAL", expand=True)[1]
        df_splited['right_manufacturer'] = df_splited['right_manufacturer'].str.split("VAL", expand=True)[1]
        df_splited['right_price'] = df_splited['right_price'].str.split("VAL", expand=True)[1]
    elif data == 'DBLP-ACM':
        df_splited.rename(columns={1: 'right_title', 2: 'right_authors', 3: 'right_venue', 4: 'right_year'}, inplace=True)
        df_splited['right_title'] = df_splited['right_title'].str.split("VAL", expand=True)[1]
        df_splited['right_authors'] = df_splited['right_authors'].str.split("VAL", expand=True)[1]
        df_splited['right_venue'] = df_splited['right_venue'].str.split("VAL", expand=True)[1]
        df_splited['right_year'] = df_splited['right_year'].str.split("VAL", expand=True)[1]
    elif data == 'DBLP-GoogleScholar':
        df_splited.rename(columns={1: 'right_title', 2: 'right_authors', 3: 'right_venue', 4: 'right_year'}, inplace=True)
        df_splited['right_title'] = df_splited['right_title'].str.split("VAL", expand=True)[1]
        df_splited['right_authors'] = df_splited['right_authors'].str.split("VAL", expand=True)[1]
        df_splited['right_venue'] = df_splited['right_venue'].str.split("VAL", expand=True)[1]
        df_splited['right_year'] = df_splited['right_year'].str.split("VAL", expand=True)[1]
    elif data == 'iTunes-Amazon':
        df_splited.rename(columns={1: 'right_Song_Name', 2: 'right_Artist_Name', 3: 'right_Album_Name', 4: 'right_Genre', 5: 'right_Price', 6:'right_CopyRight', 7:'right_Time', 8:'right_Released'}, inplace=True)
        df_splited['right_Song_Name'] = df_splited['right_Song_Name'].str.split("VAL", expand=True)[1]
        df_splited['right_Artist_Name'] = df_splited['right_Artist_Name'].str.split("VAL", expand=True)[1]
        df_splited['right_Album_Name'] = df_splited['right_Album_Name'].str.split("VAL", expand=True)[1]
        df_splited['right_Genre'] = df_splited['right_Genre'].str.split("VAL", expand=True)[1]
        df_splited['right_Price'] = df_splited['right_Price'].str.split("VAL", expand=True)[1]
        df_splited['right_CopyRight'] = df_splited['right_CopyRight'].str.split("VAL", expand=True)[1]
        df_splited['right_Time'] = df_splited['right_Time'].str.split("VAL", expand=True)[1]
        df_splited['right_Released'] = df_splited['right_Released'].str.split("VAL", expand=True)[1]
    elif data == 'Walmart-Amazon':
        df_splited.rename(columns={1: 'right_title', 2: 'right_category', 3: 'right_brand', 4: 'right_modelno', 5: 'right_price'}, inplace=True)
        df_splited['right_title'] = df_splited['right_title'].str.split("VAL", expand=True)[1]
        df_splited['right_category'] = df_splited['right_category'].str.split("VAL", expand=True)[1]
        df_splited['right_brand'] = df_splited['right_brand'].str.split("VAL", expand=True)[1]
        df_splited['right_modelno'] = df_splited['right_modelno'].str.split("VAL", expand=True)[1]
        df_splited['right_price'] = df_splited['right_price'].str.split("VAL", expand=True)[1]
    else:
        print("NEED TO IMPLEMENT CONDITION")

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


def getKey(key):
    map_data = {'Beer':'Beer_Name',
                'Amazon-Google': 'title',
                'DBLP-ACM': 'title',
                'DBLP-GoogleScholar': 'title',
                'iTunes-Amazon': 'Song_Name',
                'Walmart-Amazon': 'title'}
    return map_data.get(key)

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

    column_key = getKey(data)

    initial_pairs = [(a.__getattribute__('left_' + column_key), a.__getattribute__('right_' + column_key), a.match_score, util.pair_is_protected(a, data, False))
                     for a in preds.itertuples(index=False)] #Ditto

    clusters, nextProtected = fumc.run_steraming(initial_pairs, nextProtected, k_results)
    print("\nclustering results:\n", clusters)


    # # Write clusters to json file
    # methods.clusters_to_json(clusters)
    # # Write preds to json file
    # methods.preds_to_json(data_path)

    return clusters, preds, nextProtected

def cartesian_product(source, target):
    format_source = []
    for row_source in source.to_dict(orient='records'):
        content_source = ""
        for attr_source in row_source.keys():
            content_source += 'COL %s VAL %s ' % (attr_source, row_source[attr_source])
        format_source.append(content_source)

    format_target = []
    for row_target in target.to_dict(orient='records'):
        content_target = ""
        for attr_target in row_target.keys():
            content_target += 'COL %s VAL %s ' % (attr_target, row_target[attr_target])
        format_target.append(content_target)

    #return the pairs (cartesian among the lines)
    return [[s,t] for s in format_source for t in format_target]


def run_matching_ranking_streaming(data, list_of_pairs, nextProtected, k_results, config, model, threshold, summarizer, dk_injector, lm, ranking_mode, groups_data_assessment):
    ###########
    # Matching with Ditto
    ###########
    print("--- Matching with Ditto ...  ---")
    path_matches = "output/output_small_test_fair.jsonl"
    #organizing the pairs (to be compared)
    # input_dataframe = cartesian_product(source, target)

    #sending the pairs to be matched
    start_time = time.time()
    matcher.run_matcher_streaming2(input_dataframe=list_of_pairs, output_path=path_matches, config=config, model=model, threshold=threshold, summarizer=summarizer, dk_injector=dk_injector, lm=lm) #, lm="roberta", checkpoint_path="checkpoints/")
    time_to_match = time.time() - start_time
    print('Time to match: ' + str(time_to_match))


    ###########
    # Ranking
    ###########
    print("--- SAFER Ranking ...  ---")
    start_time = time.time()
    preds = open_ditto_result(path_matches, data)
    clusters = []

    if len(preds) > 0:
        preds = preds.sort_values(by='match_score', ascending=False)
        groups_data_assessment = []
        column_key = getKey(data)
        if not groups_data_assessment: #if groups_data_assessment is empty
            initial_pairs = [(a.__getattribute__('left_' + column_key), a.__getattribute__('right_' + column_key),
                              a.match_score, (util.pair_is_protected_by_group(a, data, False) if ranking_mode == 'm-fair' or ranking_mode == 'none'
                                              else util.pair_is_protected(a, data, False)))
                             for a in preds.itertuples(index=False)] #Ditto
        # else: #data_assessment is active
        #     initial_pairs = [(a.__getattribute__('left_' + column_key), a.__getattribute__('right_' + column_key),
        #                       a.match_score, data_distribution_assessment.data_assessment_menager.define_group_index(groups_data_assessment, a, data))
        #                      for a in preds.itertuples(index=False)] #Ditto

        if ranking_mode == 'none':
            print("Skipping Ranking Step!")
            clusters = initial_pairs[0:k_results]
        elif ranking_mode == 'm-fair':
            print("Running m-fair")
            clusters = fumc.run_steraming_ranking_by_groups(initial_pairs, 1, k_results) #1 = the first protected group
        else: #default fair-er ranking
            print("Running fair-er")
            clusters = fumc.run_steraming(initial_pairs, True, k_results) #True = first the protected group

        print("\nclustering results:\n", clusters)
        #return clusters, preds, nextProtected

    else:
        print("\nNo matches detected.\n")

    time_to_rank = time.time() - start_time
    print('Time to rank: ' + str(time_to_rank))

    return clusters, preds, nextProtected, time_to_match, time_to_rank

def fetch_edge(batch):
    edges = []
    types = []
    for ex in batch:
        type = ex["type"]
        center_id = ex["center"][0]
        neighbors = []
        if "neighbors_mask" in ex:
            for i, n in enumerate(ex["neighbors"]):
                if ex["neighbors_mask"][i] == 0:
                    continue
                neighbors.append(n)
        else:
            neighbors = ex["neighbors"]
        if type == 'l':
            edges += [[center_id, n[0]] for n in neighbors]
            types += [0] * len(neighbors)
        elif type == 'r':
            edges += [[n[0], center_id] for n in neighbors]
            types += [1] * len(neighbors)
        else:
            raise NotImplementedError
    return edges, types


def compute_predictions(edges, prediction_values, scores, labels):
    df = pd.DataFrame()

    if len(prediction_values) == 0:
        return df

    for i in range(len(edges)-1):
        if int(prediction_values[i]) == 1: #only the candidates predicted as match will be consider
            # df = pd.concat([df,format_gnem_output_to_df(edges[i], scores[i], labels[i])])
            df = df.append(format_gnem_output_to_df(edges[i], scores[i], labels[i]), ignore_index=True)
    return df


def run_matching_gnem_ranking_streaming(data, list_of_pairs, nextProtected, k_results, model, embed_model, criterion, ranking_mode):
    batch = data_sender.put_in_gnem_input_form(list_of_pairs)
    ###########
    # Matching with GNEM
    ###########
    print("--- Matching with GNEM ...  ---")

    #sending the pairs to be matched
    start_time = time.time()
    model.eval()
    embed_model.eval()

    j=0
    with torch.no_grad():
        edge,type = fetch_edge(batch)
        feature, A, label, masks = embed_model(batch)
        masks = masks.view(-1)
        label = label.view(-1)[masks == 1].long()
        pred = model(feature, A)
        pred = pred[masks == 1]
        loss = criterion(pred, label)
        pred = F.softmax(pred, dim=1)

        assert pred.shape[0] == label.shape[0]
        scores = pred[:,1].detach().cpu().numpy().tolist()
        edges = edge
        labels = label.detach().cpu().numpy().tolist()
        types = type
        prediction_values = (torch.argmax(pred, dim=1).long()).detach().cpu().numpy().tolist()

    preds = compute_predictions(edges, prediction_values, scores, labels)
        # print(edges)
        # print('========')
        # print(scores)
        # print('===++===')
        # print(labels)
        # print('===--===')
        # print(prediction_values)
        # print( '===**===')
    time_to_match = time.time() - start_time
    print('Time to match: ' + str(time_to_match))


    ###########
    # Ranking
    ###########
    print("--- SAFER Ranking ...  ---")
    start_time = time.time()
    clusters = []

    print(preds.columns)
    if len(preds) > 0:
        preds = preds.sort_values(by='match_score', ascending=False)

        column_key = getKey(data)

        initial_pairs = [(a.__getattribute__('left_' + column_key), a.__getattribute__('right_' + column_key),
                          a.match_score, (util.pair_is_protected_by_group(a, data, False) if ranking_mode == 'm-fair' or ranking_mode == 'none'
                                          else util.pair_is_protected(a, data, False)))
                         for a in preds.itertuples(index=False)] #Ditto

        if ranking_mode == 'none':
            print("Skipping Ranking Step!")
            clusters = initial_pairs[0:k_results]
        elif ranking_mode == 'm-fair':
            print("Running m-fair")
            clusters = fumc.run_steraming_ranking_by_groups(initial_pairs, 1, k_results) #1 = the first protected group
        else: #default fair-er ranking
            print("Running fair-er")
            clusters = fumc.run_steraming(initial_pairs, True, k_results) #True = first the protected group

        print("\nclustering results:\n", clusters)
        #return clusters, preds, nextProtected

    else:
        print("\nNo matches detected.\n")

    time_to_rank = time.time() - start_time
    print('Time to rank: ' + str(time_to_rank))

    return clusters, preds, nextProtected, time_to_match, time_to_rank






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