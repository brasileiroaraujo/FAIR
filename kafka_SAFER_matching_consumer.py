from confluent_kafka import Consumer
import pandas as pd
import json
import evaluation.accuracy as eval
import time
from streaming.controller_fairER_streaming import match_streaming
from streaming.controller_fairER_streaming import match_rank_streaming

################

cs=Consumer({'bootstrap.servers':'localhost:9092','group.id':'python-consumer','auto.offset.reset':'earliest'})

print('Available topics to consume: ', cs.list_topics().topics)

cs.subscribe(['safer'])


# ct=Consumer({'bootstrap.servers':'localhost:9092','group.id':'python-consumer','auto.offset.reset':'earliest'})
#
# print('Available topics to consume: ', ct.list_topics().topics)
#
# ct.subscribe(['target'])
################


def merge_clusters(clusters, incremental_clusters):
    incremental_clusters.extend(clusters[0])


def main():
    list_of_pairs = []
    nextProtected = True
    k_batch = 30
    incremental_clusters = []

    while True:
        msg_s=cs.poll(1.0) #timeout
        # msg_t=ct.poll(1.0)
        if msg_s is None:
            continue
        if msg_s.error():
            print('Error: {}'.format(msg_s.error()))
            continue

        list_of_pairs.append(msg_s.value().decode('utf-8').split("SEPARATOR"))

        if (len(list_of_pairs) == k_batch):
            # print('row source:')
            # print(list_of_pairs)

            clusters, preds, av_time, nextProtected = match_rank_streaming('Beer', list_of_pairs, nextProtected)
            merge_clusters(clusters, incremental_clusters)
            list_of_pairs = []
            # current_dataframe_source.drop(current_dataframe_source.index, inplace=True)

            print('clusters:')
            print(incremental_clusters)


    cs.close()

if __name__ == '__main__':
    main()