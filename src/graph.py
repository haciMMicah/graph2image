import numpy as np
import csv
import networkx as nx


class Graph:
    def __init__(self):
        self.adjMatrix = np.array([], dtype=float, ndmin=2)
        self.numNodes = 0
        self.numEdges = 0
        self.nodeNames = []
        self.nodeInDegrees = np.array([], dtype=int, ndmin=1)
        self.nodeOutDegrees = np.array([], dtype=int, ndmin=1)
        self.colors = None

    def read_file(self, filename='', delim=','):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=delim)
            # Pop the header line and use its length to determine our adj matrix dimensions
            header = next(reader)
            self.numNodes = len(header) - 1
            self.adjMatrix = np.zeros((self.numNodes, self.numNodes), dtype=float)
            # Iterate over the rows in the csv file and populate our graph members
            for idx, row in enumerate(reader):
                for item in row[1:]:
                    if float(item) > 0:
                        self.numEdges += 1
                self.nodeNames.append(row[0])
                self.adjMatrix[:, idx] = np.array(row[1:])
            # In degrees are non-zero occurrences across the columns
            self.nodeInDegrees = np.count_nonzero(self.adjMatrix, axis=1)
            # Out degrees are non-zero occurrences across the rows
            self.nodeOutDegrees = np.count_nonzero(self.adjMatrix, axis=0)

    def read_colors(self, filename):
        self.colors = {}
        g = nx.read_graphml(filename)
        for node, data in g.nodes(data=True):
            if "color" in data:
                self.colors[node] = (int(data['r']), int(data['g']), int(data['b']))


if __name__ == "__main__":
    fname = '../resources/WitcherNetwork.csv'
    graph = Graph()
    graph.read_file(fname, delim=';')
    print(graph.nodeOutDegrees)
    print(graph.nodeInDegrees)
