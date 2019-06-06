import leidenalg

import networkx as nx
import numpy as np

from copy import deepcopy
from math import log
from igraph import *

np.random.seed(1337)


def compute_index(links, index_func, G, G_igraph):
    """ Compute index for links, provided in `links`.

    Parameters
    ----------
    links: iterable (set, list, ...)
        Links for which index should be computed

    index_func: function
        A function which assigns a score to a link. Should take 3 arguments: link, a NetworkX
        graph and an iGraph graph

    G: nx.DiGraph
        Graph in NetworkX structure

    G_igraph: ig.Graph
        Graph in iGraph structure

    Returns
    -------
    scores: list
        Computed scores for the items provided in links
    """
    scores = []
    for link in links:
        scores.append(index_func(link, G, G_igraph))
    return scores


def pref_index(link, G, G_igraph):
    """ Preferential attachment index. """
    # Multiply with -1 to "invert" classifier decision (to avoid invalid AUC)
    return - G.out_degree(link[0]) * G.out_degree(link[1])


def adamic_adar_index(link, G, G_igraph):
    """ Adamic-Adar index. """
    return sum([1 / float(log(G.degree(neighbor)))
                for neighbor in (set(nx.neighbors(G, link[0])) & set(nx.neighbors(G, link[1])))])


def baseline_index(link, G, G_igraph):
    """ Baseline index, which predicts 1 if both endpoints of a link have an in-degree of 0,
    and 1 otherwise.
    """
    if G.in_degree(link[0]) == 0 and G.in_degree(link[1]) == 0:
        return 1.0
    else:
        return 0.0


leiden_partitions = {}


def leiden_index(link, G_nx, G):
    """ Community index, based on communities, detected by Leiden algorithm. """
    global leiden_partitions
    current_graph_id = hash(G_nx.number_of_nodes())
    if current_graph_id not in leiden_partitions:
        leiden_partitions[current_graph_id] = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition)
    u = G.vs.find(label=link[0])
    v = G.vs.find(label=link[1])

    if leiden_partitions[current_graph_id].membership[u.index] == \
            leiden_partitions[current_graph_id].membership[v.index]:
        nc = leiden_partitions[current_graph_id].size(
            leiden_partitions[current_graph_id].membership[u.index])
        mc = leiden_partitions[current_graph_id].total_weight_from_comm(
            leiden_partitions[current_graph_id].membership[u.index])

        return mc / (nc * (nc - 1) / 2)
    else:
        return 0


def calculate_auc(Ln_scores, Lp_scores):
    """ Compares how often a randomly chosen positive example has a higher score than a randomly
    chosen negative example (which is what we want from a link prediction index ideally).
    Gives +1 when that is achieved and +0.5 when the scores for pos. and neg. examples are equal.
    """
    Ln_scores_with_rep = np.random.choice(list(Ln_scores), size=len(Ln_scores), replace=True)
    Lp_scores_with_rep = np.random.choice(list(Lp_scores), size=len(Lp_scores), replace=True)
    m_ = 0
    m__ = 0

    for lnval, lpval in zip(Ln_scores_with_rep, Lp_scores_with_rep):
        if lnval < lpval:
            m_ += 1
        elif lnval == lpval:
            m__ += 1

    return (m_ + m__/2) / len(Ln_scores)


def calculate_precision(Ln_scores, Lp_scores, decision_func):
    # `Ln_scores` and `Lp_scores` are (possibly unnormalized) scores returned by methods
    # `decision_func` is a function that takes a score and returns 1 if score represents a positive
    # example and 0 otherwise

    # 1 = pos., 0 = neg. classification
    Lp_cls = [decision_func(score) for score in Lp_scores]
    Ln_cls = [decision_func(score) for score in Ln_scores]

    # Number of tps = number of 1s in list of predicted classes for actual POSITIVE examples
    tp = sum(Lp_cls)
    # Number of fps = number of 1s in list of predicted classes for actual NEGATIVE examples
    fp = sum(Ln_cls)

    if (tp + fp) == 0:
        return 0

    return tp / (tp + fp)


def calculate_recall(Ln_scores, Lp_scores, decision_func):
    # `Ln_scores` and `Lp_scores` are (possibly unnormalized) scores returned by methods
    # `decision_func` is a function that takes a score and returns 1 if score represents a positive
    # example and 0 otherwise

    # 1 = pos., 0 = neg. classification
    Lp_cls = [decision_func(score) for score in Lp_scores]

    # Number of tps = number of 1s in list of predicted classes for actual POSITIVE examples
    tp = sum(Lp_cls)

    return tp / len(Lp_cls)


def find_edges_by_episode(episode, G, op='in'):
    # If `op` is 'in', finds edges IN specified episode, else finds edges AFTER specified episode
    if op == 'in':
        effective_op = lambda curr_ep: curr_ep == episode
    else:
        effective_op = lambda curr_ep: curr_ep > episode

    res = set()
    for (killer, victim, _), edge_episode in nx.get_edge_attributes(G, 'episode').items():
        if effective_op(int(edge_episode)):
            res.add((killer, victim))
    return res


def evaluate_original_distribution(episode, num_samples, G):
    """ Evaluate methods on original (highly unbalanced) distribution.
    How this works:
    1.) Calculate density of directed network.
    2.) Sample density * num_samples positive examples that appear >= ep. `episode`
    3.) Sample `(num_samples - num_pos_ex)` negative examples from entire network.

    Parameters
    ----------
    TODO: after this function is finished
    """
    m, n = G.number_of_edges(), G.number_of_nodes()
    density = m / (n * (n - 1))
    # ceil because we want to get to sample at least 1 positive example
    num_pos_samples = min(num_samples, int(np.ceil(density * m)))
    num_neg_samples = num_samples - num_pos_samples

    # Sample positives from links after specified episode
    pos_options = sorted(find_edges_by_episode(episode - 1, G_orig, op='after'))
    pos_samples = np.random.choice(pos_options, num_pos_samples, replace=False)
    G_orig.remove_edges_from(pos_samples)

    # Sample negatives from entire network
    neg_samples = set()
    while len(neg_samples) < num_neg_samples:
        node1, node2 = np.random.choice(G_orig.nodes(), 2, replace=False)
        # node1, node2 = random.sample(G_orig.nodes(), 2)
        if node1 not in G_orig.neighbors(node2) and \
                node2 not in G_orig.neighbors(node1):
            neg_samples.add((node1, node2))

    # Sort to make results deterministic (no guaranteed order in sets/dicts)
    neg_samples = sorted(neg_samples)
    # TODO: some prediction
    print("Sampled {} positive samples and {} negative samples...".format(len(pos_samples),
                                                                          len(neg_samples)))


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

    print("Running calculations {} times ...".format(RUNS))
    predict_from_episode = 30

    for run in range(RUNS):
        print("Run {}...".format(run))
        G_full = deepcopy(G_orig)

        Lp_predictions = {'pref': [], 'aa': [], 'comm': [], 'baseline': []}
        Ln_predictions = {'pref': [], 'aa': [], 'comm': [], 'baseline': []}

        for episode in range(predict_from_episode, 60 + 1):
            G = deepcopy(G_full)
            # Sort to make results deterministic (no guaranteed order in sets/dicts)
            Lp = sorted(find_edges_by_episode(episode, G, op='in'))
            Lp_after = sorted(find_edges_by_episode(episode, G, op='after'))

            G.remove_edges_from(Lp)
            G.remove_edges_from(Lp_after)

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
            node1, node2 = np.random.choice(G_orig.nodes(), 2, replace=False)
            if node1 not in G_full.neighbors(node2) and \
                    node2 not in G_full.neighbors(node1):
                Ln.add((node1, node2))

        # Sort to make results deterministic (no guaranteed order in sets/dicts)
        Ln = sorted(Ln)

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

        # Inverting classifier decisions with function `s < 0` to get valid AUC (>= 0.5)
        pref_prec.append(calculate_precision(Ln_predictions['pref'],
                                             Lp_predictions['pref'],
                                             decision_func=(lambda s: s < 0)))
        pref_rec.append(calculate_recall(Ln_predictions['pref'],
                                         Lp_predictions['pref'],
                                         decision_func=(lambda s: s < 0)))

        adamic_adar_prec.append(calculate_precision(Ln_predictions['aa'],
                                                    Lp_predictions['aa'],
                                                    decision_func=(lambda s: s > 0)))
        adamic_adar_rec.append(calculate_recall(Ln_predictions['aa'],
                                                Lp_predictions['aa'],
                                                decision_func=(lambda s: s > 0)))

        leiden_prec.append(calculate_precision(Ln_predictions['comm'],
                                               Lp_predictions['comm'],
                                               decision_func=(lambda s: s > 0)))
        leiden_rec.append(calculate_recall(Ln_predictions['comm'],
                                           Lp_predictions['comm'],
                                           decision_func=(lambda s: s > 0)))

        random_prec.append(calculate_precision(Ln_predictions['baseline'],
                                               Lp_predictions['baseline'],
                                               decision_func=(lambda s: s > 0)))
        random_rec.append(calculate_recall(Ln_predictions['baseline'],
                                           Lp_predictions['baseline'],
                                           decision_func=(lambda s: s > 0)))

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
    print("Recall (Adamic-Adar index)")
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
