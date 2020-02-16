import networkx as nx
import numpy as np
from node2vec import Node2Vec

if __name__ == "__main__":
    EMBEDDING_SIZE = 128
    got_social_network = nx.read_graphml("data/got-network.graphml")
    # TODO: set parameters
    n2v = Node2Vec(got_social_network, dimensions=EMBEDDING_SIZE, walk_length=30, num_walks=200, workers=4)
    model = n2v.fit(window=10, min_count=1)
    # Vector that is assigned to unseen nodes
    UNK_EMBEDDING = np.random.random(EMBEDDING_SIZE)

    G = nx.read_pajek("data/deaths.net")
    for n1, n2, _ in list(G.edges):
        num_unk = 0
        try:
            n1_emb = model.wv[n1]
        except KeyError:
            n1_emb = UNK_EMBEDDING
            num_unk += 1

        try:
            n2_emb = model.wv[n2]
        except KeyError:
            n2_emb = UNK_EMBEDDING
            num_unk += 1

        # TODO: same characters are written differently in `got-network.graphml`, need to remap names
        #       (otherwise there are a lot of unknown nodes and we're essentially guessing randomly)
        print(f"{n1}-{n2}-{num_unk}x UNK")
        link_emb = 0.5 * (n1_emb + n2_emb)
        print(link_emb)


