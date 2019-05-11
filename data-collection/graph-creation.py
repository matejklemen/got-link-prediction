import networkx as nx
import pandas as pd

df = pd.read_csv("../data/deaths.csv")
print(df.head())
g = nx.DiGraph()

for index, row in df.iterrows():
    g.add_edge(row[" Killed by"].strip(), row["Character"].strip())

nx.write_adjlist(g, "../data/deaths.adj")
nx.write_pajek(g, "../data/deaths.net")
nx.write_gml(g, "../data/deaths.gml")

print("Number of edges: " + str(g.number_of_edges()))
print("Number of nodes: " + str(g.number_of_nodes()))
