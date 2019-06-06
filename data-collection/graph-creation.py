import networkx as nx
import pandas as pd

df = pd.read_csv("../data/deaths.csv")
print(df.head())
g = nx.DiGraph()

episodes = {}

already_added_characters = set()

for index, row in df.iterrows():
    u = row[" Killed by"].strip()
    v = row["Character"].strip()
    episode = row[" Episode ID"]

    # add edge
    g.add_edge(u, v)
    already_added_characters.add(u)
    already_added_characters.add(v)

    # save episode metadata
    episodes[(u, v)] = {
        'episode': str(episode)
    }

# apply episode metadata to all edges of the graph
nx.set_edge_attributes(g, episodes)

social_g = nx.read_graphml('../data/got-network.graphml')

for node in social_g.nodes():
    for already_added_character in list(already_added_characters):
        if node == already_added_character or already_added_character.startswith(node):
            continue
        else:
            g.add_node(node)

nx.write_adjlist(g, "../data/deaths.adj")
nx.write_pajek(g, "../data/deaths.net")
nx.write_gml(g, "../data/deaths.gml")

print("Number of edges: " + str(g.number_of_edges()))
print("Number of nodes: " + str(g.number_of_nodes()))
