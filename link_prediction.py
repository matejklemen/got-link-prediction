import networkx as nx
import random
from math import log
import leidenalg
from igraph import *
from math import factorial as fac
import numpy as np
from copy import deepcopy

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
    return G.out_degree(link[0]) * G.out_degree(link[1])


def adamic_adar_index(link, G, G_igraph):
    return sum([1 / float(log(G.degree(neighbor)))
                for neighbor in (set(nx.neighbors(G, link[0])) & set(nx.neighbors(G, link[1])))])


def baseline_index(link, G, G_igraph):
    if G.in_degree(link[0]) == 0 and G.in_degree(link[1]) == 0:
        return 1
    else:
        return 0


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


def calculate_auc(Ln_scores, Lp_scores):
    Ln_scores_with_rep = random.choices(
        list(Ln_scores), k=round(len(Ln_scores)))
    Lp_scores_with_rep = random.choices(
        list(Lp_scores), k=round(len(Lp_scores)))
    m_ = 0
    m__ = 0

    for lnval, lpval in zip(Ln_scores_with_rep, Lp_scores_with_rep):
        if lnval < lpval:
            m_ += 1
        elif lnval == lpval:
            m__ += 1

    return (m_ + m__/2) / len(Ln_scores)


def calculate_precision(Ln_scores, Lp_scores, thresh):
    # `Ln_scores` and `Lp_scores` are (possibly unnormalized) scores returned by methods
    # `thresh` is a threshold - scores > this threshold are classified as positive, scores
    # below this threshold are classified as negative

    # 1 = pos., 0 = neg. classification
    Lp_cls = [int(score > thresh) for score in Lp_scores]
    Ln_cls = [int(score > thresh) for score in Ln_scores]

    # Number of tps = number of 1s in list of predicted classes for actual POSITIVE examples
    tp = sum(Lp_cls)
    # Number of fps = number of 1s in list of predicted classes for actual NEGATIVE examples
    fp = sum(Ln_cls)

    if (tp + fp) == 0:
        return 0

    return tp / (tp + fp)


def calculate_recall(Ln_scores, Lp_scores, thresh):
    # `Ln_scores` and `Lp_scores` are (possibly unnormalized) scores returned by methods
    # `thresh` is a threshold - scores > this threshold are classified as positive, scores
    # below this threshold are classified as negative

    # 1 = pos., 0 = neg. classification
    Lp_cls = [int(score > thresh) for score in Lp_scores]

    # Number of tps = number of 1s in list of predicted classes for actual POSITIVE examples
    tp = sum(Lp_cls)

    return tp / len(Lp_cls)


def find_edges_after_episode(episode, G):
    res = set()
    for edge, edge_episode in nx.get_edge_attributes(G, 'episode').items():
        if int(edge_episode) > episode:
            res.add(edge)
    return res


def find_edges_in_episode(episode, G):
    res = set()
    for edge, edge_episode in nx.get_edge_attributes(G, 'episode').items():
        if int(edge_episode) == episode:
            res.add(edge)
    return res


if __name__ == "__main__":
    RUNS = 5
    G_orig = nx.read_pajek('./data/deaths.net')

    m = G_orig.number_of_edges()
    # AUCs over several runs
    pref_scores = []
    adamic_adar_scores = []
    leiden_scores = []
    random_scores = []

    # Precision and recall over several runs
    pref_prec, pref_rec = [], []
    adamic_adar_prec, adamic_adar_rec = [], []
    leiden_prec, leiden_rec = [], []
    random_prec, random_rec = [], []

    print("Running calculations " + str(RUNS) + " times ...")

    predict_from_episode = 50

    for run in range(RUNS):
        print("Run: ", run)
        G_full = deepcopy(G_orig)

        Lp_predictions = {'pref': [], 'aa': [], 'comm': [], 'baseline': []}
        Ln_predictions = {'pref': [], 'aa': [], 'comm': [], 'baseline': []}

        for episode in range(predict_from_episode, 60 + 1):
            G = deepcopy(G_full)
            Lp = find_edges_in_episode(episode, G)
            Lp_after = find_edges_after_episode(episode, G)

            G.remove_edges_from(list(Lp))
            G.remove_edges_from(list(Lp_after))

            # sending the adjusted graph to iGraph (sorry for hacks)
            nx.write_gml(G, './data/deaths_removededges.gml')
            G_igraph = Graph.Read_GML('./data/deaths_removededges.gml')

            Lp_predictions['pref'].extend(
                compute_index(Lp, pref_index, G, G_igraph))
            Lp_predictions['aa'].extend(compute_index(
                Lp, adamic_adar_index, G, G_igraph))
            Lp_predictions['comm'].extend(
                compute_index(Lp, leiden_index, G, G_igraph))
            Lp_predictions['baseline'].extend(
                compute_index(Lp, baseline_index, G, G_igraph))

        # sending the full graph to iGraph
        nx.write_gml(G_full, './data/deaths_removededges.gml')
        G_igraph = Graph.Read_GML('./data/deaths_removededges.gml')

        Ln = set()
        while len(Ln) < len(Lp_predictions['pref']):
            node1, node2 = random.sample(G_full.nodes(), 2)
            if node1 not in G_full.neighbors(node2) and \
                    node2 not in G_full.neighbors(node1):
                Ln.add((node1, node2))

        Ln_predictions['pref'] = compute_index(
            Ln, pref_index, G_full, G_igraph)
        Ln_predictions['aa'] = compute_index(
            Ln, adamic_adar_index, G_full, G_igraph)
        Ln_predictions['comm'] = compute_index(
            Ln, leiden_index, G_full, G_igraph)
        Ln_predictions['baseline'] = compute_index(
            Ln, baseline_index, G_full, G_igraph)

        pref_scores.append(calculate_auc(
            Ln_predictions['pref'], Lp_predictions['pref']))
        adamic_adar_scores.append(calculate_auc(
            Ln_predictions['aa'], Lp_predictions['aa']))
        leiden_scores.append(calculate_auc(
            Ln_predictions['comm'], Lp_predictions['comm']))
        random_scores.append(calculate_auc(
            Ln_predictions['baseline'], Lp_predictions['baseline']))

        pref_prec.append(calculate_precision(Ln_predictions['pref'],
                                             Lp_predictions['pref'],
                                             thresh=0))
        pref_rec.append(calculate_recall(Ln_predictions['pref'],
                                         Lp_predictions['pref'],
                                         thresh=0))

        adamic_adar_prec.append(calculate_precision(Ln_predictions['aa'],
                                                    Lp_predictions['aa'],
                                                    thresh=0))
        adamic_adar_rec.append(calculate_recall(Ln_predictions['aa'],
                                                Lp_predictions['aa'],
                                                thresh=0))

        leiden_prec.append(calculate_precision(Ln_predictions['comm'],
                                               Lp_predictions['comm'],
                                               thresh=0))
        leiden_rec.append(calculate_recall(Ln_predictions['comm'],
                                           Lp_predictions['comm'],
                                           thresh=0))

        random_prec.append(calculate_precision(Ln_predictions['baseline'],
                                               Lp_predictions['baseline'],
                                               thresh=0))
        random_rec.append(calculate_recall(Ln_predictions['baseline'],
                                           Lp_predictions['baseline'],
                                           thresh=0))

    # Print mean results with the standard deviation for all indices
    print("\n----")
    print("AUC (Preferential attachment index)")
    print("\t - Mean:", np.mean(pref_scores))
    print("\t - Std. deviation:", np.std(pref_scores))
    print("Precision (Preferential attachment index)")
    print("\t - Mean:", np.mean(pref_prec))
    print("\t - Std. deviation:", np.std(pref_prec))
    print("Recall (Preferential attachment index)")
    print("\t - Mean:", np.mean(pref_rec))
    print("\t - Std. deviation:", np.std(pref_rec))
    print("----")

    print("\nAUC (Adamic-Adar index)")
    print("\t - Mean:", np.mean(adamic_adar_scores))
    print("\t - Std. deviation:", np.std(adamic_adar_scores))
    print("Precision (Adamic-Adar index)")
    print("\t - Mean:", np.mean(adamic_adar_prec))
    print("\t - Std. deviation:", np.std(adamic_adar_prec))
    print("\nRecall (Adamic-Adar index)")
    print("\t - Mean:", np.mean(adamic_adar_rec))
    print("\t - Std. deviation:", np.std(adamic_adar_rec))
    print("----")

    print("\nAUC (Community index)")
    print("\t - Mean:", np.mean(leiden_scores))
    print("\t - Std. deviation:", np.std(leiden_scores))
    print("Precision (Community index)")
    print("\t - Mean:", np.mean(leiden_prec))
    print("\t - Std. deviation:", np.std(leiden_prec))
    print("Recall (Community index)")
    print("\t - Mean:", np.mean(leiden_rec))
    print("\t - Std. deviation:", np.std(leiden_rec))
    print("----")

    print("\nAUC (Random index)")
    print("\t - Mean:", np.mean(random_scores))
    print("\t - Std. deviation:", np.std(random_scores))
    print("Precision (Random index)")
    print("\t - Mean:", np.mean(random_prec))
    print("\t - Std. deviation:", np.std(random_prec))
    print("Recall (Random index)")
    print("\t - Mean:", np.mean(random_rec))
    print("\t - Std. deviation:", np.std(random_rec))
    print("----")
