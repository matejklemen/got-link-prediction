import networkx as nx
import numpy as np
from node2vec import Node2Vec
import json

if __name__ == "__main__":
    EMBEDDING_SIZE = 128
    got_social_network = nx.read_graphml("data/got-network.graphml")
    # TODO: set parameters
    n2v = Node2Vec(got_social_network, dimensions=EMBEDDING_SIZE, walk_length=30, num_walks=200, workers=4)
    model = n2v.fit(window=10, min_count=1)
    # Vector that is assigned to unseen nodes
    UNK_EMBEDDING = np.random.random(EMBEDDING_SIZE)

    num_unks = 0
    with open("data/name_synonyms.json") as f:
        name_map = json.load(f)
    G = nx.read_pajek("data/deaths.net")
    for n1, n2, _ in list(G.edges):
        n1, n2 = name_map.get(n1, n1), name_map.get(n2, n2)
        try:
            n1_emb = model.wv[n1]
        except KeyError:
            n1_emb = UNK_EMBEDDING

        try:
            n2_emb = model.wv[n2]
        except KeyError:
            n2_emb = UNK_EMBEDDING

        link_emb = 0.5 * (n1_emb + n2_emb)
        num_unks += int(np.all(link_emb == UNK_EMBEDDING))

    print(f"{num_unks} links have a random embedding (out of {len(G.edges) * 2})")
