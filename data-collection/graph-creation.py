import networkx as nx
import pandas as pd

df = pd.read_csv("../data/deaths.csv")
print(df.head())
g = nx.DiGraph()

episodes = {}

for index, row in df.iterrows():
    u = row[" Killed by"].strip()
    v = row["Character"].strip()
    episode = row[" Episode ID"]

    # add edge
    g.add_edge(u, v)

    # save episode metadata
    episodes[(u, v)] = {
        'episode': str(episode)
    }

# apply episode metadata to all edges of the graph
nx.set_edge_attributes(g, episodes)

nx.write_adjlist(g, "../data/deaths.adj")
nx.write_pajek(g, "../data/deaths.net")
nx.write_gml(g, "../data/deaths.gml")

print("Number of edges: " + str(g.number_of_edges()))
print("Number of nodes: " + str(g.number_of_nodes()))
