import leidenalg

import networkx as nx
import numpy as np
import pandas as pd

from copy import deepcopy
from math import log
from igraph import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

np.random.seed(1337)

net_feats = pd.read_csv('./data/additional_character_features.csv')
pagerank_mean = net_feats['PageRank'].mean()
betweenness_mean = net_feats['Betweenness'].mean()
dummy_community = net_feats['Community'].max() + 1


def display_results(name, auc_runs, prec_runs, rec_runs):
    print(f"{name}:")
    print(f"[AUC] {np.mean(auc_runs):.3f} +- {np.std(auc_runs):.3f}", end=" | ")
    print(f"[prec.] {np.mean(prec_runs):.3f} +- {np.std(prec_runs):.3f}", end=" | ")
    print(f"[rec.] {np.mean(rec_runs):.3f} +- {np.std(rec_runs):.3f}")


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


def pref_index(link, G, G_igraph, use_death_info=False):
    """ Preferential attachment index. If `use_death_info` is True, additional check will be performed prior to
    index calculation: if the alleged perpetrator and alleged victim are both dead, the index score will be -inf.

    Higher score implies higher likelihood of a positive outcome.
    """
    if use_death_info:
        # src dead and/or dst dead => src can't kill dst
        if G.in_degree(link[0]) > 0 or G.in_degree(link[1]) > 0:
            return -float("inf")

    return G.out_degree(link[0]) * G.out_degree(link[1])


def adamic_adar_index(link, G, G_igraph, use_death_info=False):
    """ Adamic-Adar index. If `use_death_info` is True, additional check will be performed prior to index calculation:
    if the alleged perpetrator and alleged victim are both dead, the index will be -inf.

    Higher score implies higher likelihood of a positive outcome.
    """
    if use_death_info:
        # src dead and/or dst dead => src can't kill dst
        if G.in_degree(link[0]) > 0 or G.in_degree(link[1]) > 0:
            return -float("inf")

    return sum([1 / float(log(G.degree(neighbor)))
                for neighbor in (set(nx.neighbors(G, link[0])) & set(nx.neighbors(G, link[1])))])


def baseline_index(link, G, G_igraph):
    """ Baseline index, which predicts 1.0 if both endpoints of a link have an in-degree of 0,
    and 0.0 otherwise.
    """
    if G.in_degree(link[0]) == 0 and G.in_degree(link[1]) == 0:
        return 1.0
    else:
        return 0.0 - 10e-5


leiden_partitions = {}


def leiden_index(link, G_nx, G, use_death_info=False):
    """ Community index, based on communities, detected by Leiden algorithm.

    Higher score implies higher likelihood of a positive outcome.
    """
    if use_death_info:
        # src dead and/or dst dead => src can't kill dst
        if G_nx.in_degree(link[0]) > 0 or G_nx.in_degree(link[1]) > 0:
            return -float("inf")

    global leiden_partitions
    current_graph_id = hash(G_nx.number_of_nodes())
    if current_graph_id not in leiden_partitions:
        leiden_partitions[current_graph_id] = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition)
    u = G.vs.find(label=link[0])
    v = G.vs.find(label=link[1])

    u_community = leiden_partitions[current_graph_id].membership[u.index]
    v_community = leiden_partitions[current_graph_id].membership[v.index]

    if u_community == v_community:
        nc = leiden_partitions[current_graph_id].size(u_community)
        mc = leiden_partitions[current_graph_id].total_weight_from_comm(u_community)

        return mc / (nc * (nc - 1) / 2)
    else:
        # count links between communities
        nodes_in_u_community = [i for i, x in enumerate(leiden_partitions[current_graph_id].membership) if x == u_community]
        nodes_in_v_community = {i for i, x in enumerate(leiden_partitions[current_graph_id].membership) if x == v_community}

        links_between_communities = 0
        for node in nodes_in_u_community:
            for edge_id in G.incident(G.vs[node]):
                _, target = G.es[edge_id].tuple
                if target in nodes_in_v_community:
                    links_between_communities += 1

        return links_between_communities / (len(nodes_in_u_community) * len(nodes_in_v_community))


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
    """ Calculate precision = |tp| / (|tp| + |fp|)

    Parameters
    ----------
    decision_func: function
        Function that converts scores into 0/1 (neg/pos) outcome (e.g. `lambda score: score > 0.5`)
    """
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
    """ Calculate recall = |tp| / (|tp| + |fn|) = |tp| / |pos|

    Parameters
    ----------
    decision_func: function
        Function that converts scores into 0/1 (neg/pos) outcome (e.g. `lambda score: score > 0.5`)
    """
    # Number of TPs = number of 1s in list of predicted classes for actual POSITIVE examples
    Lp_cls = [decision_func(score) for score in Lp_scores]
    tp = sum(Lp_cls)

    return tp / len(Lp_cls)


def find_edges_by_episode(episode, G, op='in'):
    # If `op` is 'in', finds edges IN specified episode,
    # if `op` is 'before', finds edges BEFORE specified episode,
    # else finds edges AFTER specified episode
    if op == 'in':
        effective_op = lambda curr_ep: curr_ep == episode
    elif op == 'before':
        effective_op = lambda curr_ep: curr_ep < episode
    else:
        effective_op = lambda curr_ep: curr_ep > episode

    res = set()
    for (killer, victim, _), edge_episode in nx.get_edge_attributes(G, 'episode').items():
        if effective_op(int(edge_episode)):
            res.add((killer, victim))
    return res


def sample_negative_examples(G, num_neg_samples):
    # Sample negatives from entire network
    neg_samples = set()
    while len(neg_samples) < num_neg_samples:
        node1, node2 = np.random.choice(G.nodes(), 2, replace=False)
        if node1 not in nx.all_neighbors(G, node2) and \
                node2 not in nx.all_neighbors(G, node1):
            neg_samples.add((node1, node2))

    return neg_samples


def get_additional_features(character):
    """
    Gets additional features (PageRank, Betweenness, Community)
    from the additional_character_features.csv file, computed on the social network
    of Game of Thrones characters from the book.
    """
    for short_name in net_feats['Character']:
        if character.startswith(short_name):
            # if the short_name is the start of the full character's name
            # we can assume that it is the same character
            character_row = net_feats.loc[net_feats['Character'] == short_name]
            return float(character_row['PageRank']), float(character_row['Betweenness']), int(character_row['Community'])

    # no character found in the additional features dataset, we use the means and the most popular community
    return pagerank_mean, betweenness_mean, dummy_community
    

def extract_features(G, edge):
    u, v = edge[0], edge[1]
    u_pagerank, u_betweenness, u_community = get_additional_features(u)
    v_pagerank, v_betweenness, v_community = get_additional_features(v)
    
    return [G.out_degree(u), G.out_degree(v), u_pagerank, u_betweenness, u_community, v_pagerank, v_betweenness, v_community]


def ml_approach(G, episode, models=None):
    """ How this works:
    [1. Prediction for positive samples]
    - set each episode in range [episode, 60] as threshold for getting examples
        - take kills from episode that is currently set as thresh (= current TEST SET)
        - take kills from episodes prior to the current thresh
        - sample as many negative examples as there are kills obtained in previous step
        - this way we get a balanced (50% kills, 50% non-kills) training set

    [2. Prediction for negative samples]
    - take all kills in the network
    - sample the same amount of negative examples
    - again, we have a balanced training set
    - test set here contains as many negative examples as there were positive examples in [1.]

    Note that multiple models can be evaluated on these examples in one run of function
    (to make sure that some model does not get a lucky break and get a higher score that way).

    Parameters:
        models: List of instances of models, on which we want to evaluate the approach.
                For example, using logistic regression and SVM:
                >>> m1, m2 = LogisticRegression(), SVC()
                >>> ml_approach(..., ..., [m1, m2])

    Returns:
        List of pairs (Lp_scores, Ln_scores) for each classifier, specified in `models`.
    """
    if models is None:
        model = KNeighborsClassifier()
        models = [model]

    Lp_preds = [[] for _ in models]
    Ln_preds = [[] for _ in models]
    for curr_episode_thresh in range(episode, 60 + 1):
        G_copy = deepcopy(G)
        Lp_train = sorted(find_edges_by_episode(curr_episode_thresh, G_copy, op='before'))
        Ln_train = sorted(sample_negative_examples(G_copy, len(Lp_train)))

        Lp_test = sorted(find_edges_by_episode(curr_episode_thresh, G_copy, op='in'))
        # Episode with no kills, i.e. nothing to predict
        if len(Lp_test) == 0:
            continue
        Lp_after_ep = sorted(find_edges_by_episode(curr_episode_thresh, G_copy, op='after'))

        G_copy.remove_edges_from(Lp_test)
        G_copy.remove_edges_from(Lp_after_ep)

        # Extract features for training and test examples
        X_train = [extract_features(G_copy, curr_example) for curr_example in Lp_train]
        y_train = [1 for _ in range(len(Lp_train))]
        X_train.extend([extract_features(G_copy, curr_example) for curr_example in Ln_train])
        y_train.extend([0 for _ in range(len(Ln_train))])
        X_test = [extract_features(G_copy, curr_example) for curr_example in Lp_test]

        # Make sure we have 2D arrays (could otherwise be problematic if there's only 1 test case)
        X_train = np.atleast_2d(X_train)
        X_test = np.atleast_2d(X_test)

        for i, curr_model in enumerate(models):
            curr_model.fit(X_train, y_train)
            preds = curr_model.predict_proba(X_test)
            preds = preds[:, 1]
            Lp_preds[i].extend(preds)

    G_copy = deepcopy(G)
    Lp_train = sorted(find_edges_by_episode(60 + 1, G_copy, op='before'))
    neg_examples = sorted(sample_negative_examples(G_copy, len(Lp_train) + len(Lp_preds[0])))
    np.random.shuffle(neg_examples)
    Ln_train = neg_examples[: len(Lp_train)]
    Ln_test = neg_examples[len(Lp_train):]

    # Extract features for training and test examples
    X_train = [extract_features(G_copy, curr_example) for curr_example in Lp_train]
    y_train = [1 for _ in range(len(Lp_train))]
    X_train.extend([extract_features(G_copy, curr_example) for curr_example in Ln_train])
    y_train.extend([0 for _ in range(len(Ln_train))])
    X_test = [extract_features(G_copy, curr_example) for curr_example in Ln_test]

    for i, curr_model in enumerate(models):
        curr_model.fit(X_train, y_train)
        curr_preds = curr_model.predict_proba(X_test)
        curr_preds = curr_preds[:, 1]

        # Take probabilities of the positive class as scores -
        # if example is predicted to be positive (kill), this score will be high
        # if example is predicted to be negative (no kill), this score will be low
        Ln_preds[i].extend(curr_preds)

    Lp_Ln_preds = list(zip(Lp_preds, Ln_preds))
    return Lp_Ln_preds


if __name__ == "__main__":
    RUNS = 5
    G_orig = nx.read_pajek('./data/deaths.net')

    m = G_orig.number_of_edges()
    # AUC, precision and recall over several runs
    pref_scores, pref_prec, pref_rec = [], [], []
    prefd_scores, prefd_prec, prefd_rec = [], [], []
    adamic_adar_scores, adamic_adar_prec, adamic_adar_rec = [], [], []
    aad_scores, aad_prec, aad_rec = [], [], []
    leiden_scores, leiden_prec, leiden_rec = [], [], []
    ld_scores, ld_prec, ld_rec = [], [], []
    random_scores, random_prec, random_rec = [], [], []
    knn_scores, knn_prec, knn_rec = [], [], []
    logr_scores, logr_prec, logr_rec = [], [], []
    svm_scores, svm_prec, svm_rec = [], [], []

    print('Running calculations {} times ...'.format(RUNS))
    predict_from_episode = 30

    for run in range(RUNS):
        print('Run {}...'.format(run))
        G_full = deepcopy(G_orig)

        Lp_predictions = {'pref': [], 'pref-death': [], 'aa': [], 'aa-death': [], 'comm': [], 'comm-death': [], 'baseline': []}
        Ln_predictions = {'pref': [], 'pref-death': [], 'aa': [], 'aa-death': [], 'comm': [], 'comm-death': [], 'baseline': []}

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
            Lp_predictions['pref-death'].extend(
                compute_index(Lp, lambda link, G, G_ig: pref_index(link, G, G_ig, use_death_info=True), G, G_igraph))
            Lp_predictions['aa'].extend(compute_index(
                Lp, adamic_adar_index, G, G_igraph))
            Lp_predictions['aa-death'].extend(compute_index(
                Lp, lambda link, G, G_ig: adamic_adar_index(link, G, G_ig, use_death_info=True), G, G_igraph))
            Lp_predictions['comm'].extend(
                compute_index(Lp, leiden_index, G, G_igraph))
            Lp_predictions['comm-death'].extend(
                compute_index(Lp, lambda link, G, G_ig: leiden_index(link, G, G_igraph, use_death_info=True), G, G_igraph))
            Lp_predictions['baseline'].extend(
                compute_index(Lp, baseline_index, G, G_igraph))

        # sending the full graph to iGraph
        nx.write_gml(G_full, './data/deaths_removededges.gml')
        G_igraph = Graph.Read_GML('./data/deaths_removededges.gml')

        # Sort to make results deterministic (no guaranteed order in sets/dicts)
        Ln = sorted(sample_negative_examples(G_orig, len(Lp_predictions['pref'])))

        Ln_predictions['pref'] = compute_index(
            Ln, pref_index, G_full, G_igraph)
        Ln_predictions['pref-death'] = compute_index(
            Ln, lambda link, G, G_ig: pref_index(link, G, G_ig, use_death_info=True), G_full, G_igraph)
        Ln_predictions['aa'] = compute_index(
            Ln, adamic_adar_index, G_full, G_igraph)
        Ln_predictions['aa-death'] = compute_index(
            Ln, lambda link, G, G_ig: adamic_adar_index(link, G, G_ig, use_death_info=True), G_full, G_igraph)
        Ln_predictions['comm'] = compute_index(
            Ln, leiden_index, G_full, G_igraph)
        Ln_predictions['comm-death'] = compute_index(
            Ln, lambda link, G, G_ig: leiden_index(link, G, G_ig, use_death_info=True), G_full, G_igraph)
        Ln_predictions['baseline'] = compute_index(
            Ln, baseline_index, G_full, G_igraph)

        G_copy = deepcopy(G_orig)
        m1 = KNeighborsClassifier()
        m2 = LogisticRegression(solver='liblinear')
        m3 = SVC(probability=True, gamma='auto')

        (Lp_preds_knn, Ln_preds_knn), (Lp_preds_logr, Ln_preds_logr),  (Lp_preds_svm, Ln_preds_svm) = \
            ml_approach(G_copy, episode=predict_from_episode, models=[m1, m2, m3])

        pref_scores.append(calculate_auc(
            Ln_predictions['pref'], Lp_predictions['pref']))
        prefd_scores.append(calculate_auc(
            Ln_predictions['pref-death'], Lp_predictions['pref-death']))
        adamic_adar_scores.append(calculate_auc(
            Ln_predictions['aa'], Lp_predictions['aa']))
        aad_scores.append(calculate_auc(
            Ln_predictions['aa-death'], Lp_predictions['aa-death']))
        leiden_scores.append(calculate_auc(
            Ln_predictions['comm'], Lp_predictions['comm']))
        ld_scores.append(calculate_auc(
            Ln_predictions['comm-death'], Lp_predictions['comm-death']))
        random_scores.append(calculate_auc(
            Ln_predictions['baseline'], Lp_predictions['baseline']))
        knn_scores.append(calculate_auc(Ln_preds_knn, Lp_preds_knn))
        logr_scores.append(calculate_auc(Ln_preds_logr, Lp_preds_logr))
        svm_scores.append(calculate_auc(Ln_preds_svm, Lp_preds_svm))

        # Inverting classifier decisions with function `s < 0` to get valid AUC (>= 0.5)
        pref_prec.append(calculate_precision(Ln_predictions['pref'],
                                             Lp_predictions['pref'],
                                             decision_func=(lambda s: s > 0)))
        pref_rec.append(calculate_recall(Ln_predictions['pref'],
                                         Lp_predictions['pref'],
                                         decision_func=(lambda s: s > 0)))

        prefd_prec.append(calculate_precision(Ln_predictions['pref-death'],
                                              Lp_predictions['pref-death'],
                                              decision_func=(lambda s: s > 0)))
        prefd_rec.append(calculate_recall(Ln_predictions['pref-death'],
                                          Lp_predictions['pref-death'],
                                          decision_func=(lambda s: s > 0)))

        adamic_adar_prec.append(calculate_precision(Ln_predictions['aa'],
                                                    Lp_predictions['aa'],
                                                    decision_func=(lambda s: s > 0)))
        adamic_adar_rec.append(calculate_recall(Ln_predictions['aa'],
                                                Lp_predictions['aa'],
                                                decision_func=(lambda s: s > 0)))

        aad_prec.append(calculate_precision(Ln_predictions['aa-death'],
                                            Lp_predictions['aa-death'],
                                            decision_func=(lambda s: s > 0)))
        aad_rec.append(calculate_recall(Ln_predictions['aa-death'],
                                        Lp_predictions['aa-death'],
                                        decision_func=(lambda s: s > 0)))

        leiden_prec.append(calculate_precision(Ln_predictions['comm'],
                                               Lp_predictions['comm'],
                                               decision_func=(lambda s: s > 0)))
        leiden_rec.append(calculate_recall(Ln_predictions['comm'],
                                           Lp_predictions['comm'],
                                           decision_func=(lambda s: s > 0)))

        ld_prec.append(calculate_precision(Ln_predictions['comm-death'],
                                           Lp_predictions['comm-death'],
                                           decision_func=(lambda s: s > 0)))
        ld_rec.append(calculate_recall(Ln_predictions['comm-death'],
                                       Lp_predictions['comm-death'],
                                       decision_func=(lambda s: s > 0)))

        random_prec.append(calculate_precision(Ln_predictions['baseline'],
                                               Lp_predictions['baseline'],
                                               decision_func=(lambda s: s > 0)))
        random_rec.append(calculate_recall(Ln_predictions['baseline'],
                                           Lp_predictions['baseline'],
                                           decision_func=(lambda s: s > 0)))

        knn_prec.append(calculate_precision(Ln_preds_knn, Lp_preds_knn,
                                            decision_func=(lambda s: s > 0.5)))
        knn_rec.append(calculate_recall(Ln_preds_knn, Lp_preds_knn,
                                        decision_func=(lambda s: s > 0.5)))

        logr_prec.append(calculate_precision(Ln_preds_logr, Lp_preds_logr,
                                             decision_func=(lambda s: s > 0.5)))
        logr_rec.append(calculate_recall(Ln_preds_logr, Lp_preds_logr,
                                         decision_func=(lambda s: s > 0.5)))

        svm_prec.append(calculate_precision(Ln_preds_svm, Lp_preds_svm,
                                            decision_func=(lambda s: s > 0.5)))
        svm_rec.append(calculate_recall(Ln_preds_svm, Lp_preds_svm,
                                        decision_func=(lambda s: s > 0.5)))

    # Print mean results with the standard deviation for all indices
    display_results('Preferential attachment index', pref_scores, pref_prec, pref_rec)
    display_results('Preferential attachment index (using death info)', prefd_scores, prefd_prec, prefd_rec)
    display_results('Adamic-Adar index', adamic_adar_scores, adamic_adar_prec, adamic_adar_rec)
    display_results('Adamic-Adar index (using death info)', aad_scores, aad_prec, aad_rec)
    display_results('Community index', leiden_scores, leiden_prec, leiden_rec)
    display_results('Community index (using death info)', ld_scores, ld_prec, ld_rec)
    display_results('Random index', random_scores, random_prec, random_rec)
    display_results('ML (KNN)', knn_scores, knn_prec, knn_rec)
    display_results('ML (logistic reg.)', logr_scores, logr_prec, logr_rec)
    display_results('ML (SVM)', svm_scores, svm_prec, svm_rec)
