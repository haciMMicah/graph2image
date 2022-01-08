import numpy as np
import cv2 as cv
import graph as gr
from matplotlib import pyplot as plt


def sort_nodes(graph_obj):
    # Sort the indices
    ind = np.argsort(graph_obj.nodeOutDegrees)
    ind = np.flip(ind)
    return ind


def generate_circles(graph_obj, indices, width=800, height=600, radius_min=3, radius_max=40):
    circle_attr = np.random.randint([0, 0, 0], high=[height, width, 1], size=(indices.shape[0], 3))
    circle_attr[:, 2] = graph_obj.nodeOutDegrees[indices]
    image = np.zeros((height, width, 3), dtype='uint8')
    colors = np.random.randint([0, 0, 0], high=[255, 255, 255], size=(indices.shape[0], 3))
    for idx, row in enumerate(circle_attr):
        color = (int(colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2]))
        thickness = -1
        center = (circle_attr[idx, 1], circle_attr[idx, 0])
        radius = int(circle_attr[idx, 2] * .4)
        if radius < radius_min:
            radius = radius_min
        if radius > radius_max:
            radius = radius_max
        image = cv.circle(image, center, radius, color, thickness)
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    fname = '../resources/WitcherNetwork.csv'
    graph = gr.Graph()
    graph.read_file(fname, delim=';')
    indices = sort_nodes(graph)
    generate_circles(graph, indices)
