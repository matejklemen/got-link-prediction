import networkx as nx
import random
from math import log
import leidenalg
from igraph import *
from math import factorial as fac
import numpy as np

random.seed(1337)


def binomial(x, y):
    try:
        binom = fac(x) // fac(y) // fac(x - y)
    except ValueError:
        binom = 0
    return binom


def compute_index(links, index_func, G, G_igraph):
    scores = []
    for link in links:
        scores += [index_func(link, G, G_igraph)]
    return scores


def pref_index(link, G, G_igraph):
    return G.degree(link[0]) * G.degree(link[1])


def adamic_adar_index(link, G, G_igraph):
    return sum([1 / float(log(G.degree(neighbor))) for neighbor in nx.common_neighbors(G, link[0], link[1])])


leiden_partitions = {}


def leiden_index(link, G_nx, G):
    global leiden_partitions
    current_graph_id = hash(G_nx.number_of_nodes())
    if current_graph_id not in leiden_partitions:
        leiden_partitions[current_graph_id] = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition)
    u = G.vs.find(label=link[0])
    v = G.vs.find(label=link[1])

    if leiden_partitions[current_graph_id].membership[u.index] == leiden_partitions[current_graph_id].membership[v.index]:

        nc = leiden_partitions[current_graph_id].size(
            leiden_partitions[current_graph_id].membership[u.index])
        mc = leiden_partitions[current_graph_id].total_weight_from_comm(
            leiden_partitions[current_graph_id].membership[u.index])

        return mc / binomial(nc, 2)
    else:
        return 0


def get_auc_for_index(Ln, Lp, index_func, G, G_igraph):
    m_ = 0
    m__ = 0
    for lnval, lpval in zip(compute_index(Ln, index_func, G, G_igraph), compute_index(Lp, index_func, G, G_igraph)):
        if lnval < lpval:
            m_ += 1
        elif lnval == lpval:
            m__ += 1

    return (m_ + m__/2) / len(Ln)


def find_edges_after_episode(episode, G):
    res = set()
    for edge, edge_episode in nx.get_edge_attributes(G, 'episode').items():
        if int(edge_episode) > episode:
            res.add(edge)
    return res


if __name__ == "__main__":

    RUNS = 5
    G_orig = nx.Graph(nx.read_pajek('./data/deaths.net'))

    m = G_orig.number_of_edges()
    pref_scores = []
    adamic_adar_scores = []
    leiden_scores = []

    print("Running calculations " + str(RUNS) + " times ...")

    for run in range(RUNS):
        print("Run: ", run)
        G = G_orig.copy()

        Lp = find_edges_after_episode(50, G)

        Ln = set()
        while len(Ln) < len(Lp):
            node1, node2 = random.sample(G.nodes(), 2)
            if (node1 not in G.neighbors(node2)):
                Ln.add((node1, node2))

        G.remove_edges_from(list(Lp))

        # sending the adjusted graph to iGraph
        nx.write_gml(G, './data/deaths_removededges.gml')
        G_igraph = Graph.Read_GML('./data/deaths_removededges.gml')
        G_igraph = G_igraph.as_undirected()

        pref_scores.append(get_auc_for_index(
            Ln, Lp, pref_index, G, G_igraph))
        adamic_adar_scores.append(get_auc_for_index(
            Ln, Lp, adamic_adar_index, G, G_igraph))
        leiden_scores.append(get_auc_for_index(
            Ln, Lp, leiden_index, G, G_igraph))

    # Print mean results with the standard deviation for all indices
    print("\n----")
    print("AUC (Preferential attachment index)")
    print("\t - Mean:", np.mean(pref_scores))
    print("\t - Std. deviation:", np.std(pref_scores))
    print("\nAUC (Adamic-Adar index)")
    print("\t - Mean:", np.mean(adamic_adar_scores))
    print("\t - Std. deviation:", np.std(adamic_adar_scores))
    print("\nAUC (Community index)")
    print("\t - Mean:", np.mean(leiden_scores))
    print("\t - Std. deviation:", np.std(leiden_scores))
    print("----")
