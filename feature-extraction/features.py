import networkx as nx
from igraph import *
import leidenalg
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    G = nx.read_graphml('../data/got-network.graphml')
    nx.draw_spring(G, node_size=10)
    plt.show()

    """ PAGERANK """

    pr = list(nx.pagerank(G).items())
    pr.sort(key=lambda tup: tup[1], reverse=True)

    df_pagerank = pd.DataFrame(pr, columns = ['Character', 'PageRank'])

    """ BETWEENNESS """

    bt = list(nx.betweenness_centrality(G).items())
    bt.sort(key=lambda tup: tup[1], reverse=True)

    df_betweenness = pd.DataFrame(bt, columns = ['Character', 'Betweenness'])

    """ COMMUNITIES """

    G = Graph.Read_GraphML('../data/got-network.graphml')

    # partition graph
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

    df_community = pd.DataFrame([(node["label"], partition.membership[node.index]) for node in G.vs], columns = ['Character', 'Community'])

    all_data = df_pagerank.copy()
    all_data = all_data.merge(df_betweenness, on='Character')
    all_data = all_data.merge(df_community, on='Character')

    print(all_data.to_csv(index=False))

