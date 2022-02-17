import math

import numpy as np
import cv2 as cv
import graph as gr
import circle_pack as cp
from matplotlib import pyplot as plt
import argparse
import networkx as nx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Packs a contour image with the nodes of a graph")
    parser.add_argument("csv_file", metavar="C", type=str,
                        help="path to a csv file containing the adj matrix for a graph")
    parser.add_argument("graphml_file", metavar="G", type=str,
                        help="path to a graphml file to read the node color info")
    parser.add_argument("image", metavar="I", type=str, help="path to an image file of contours to pack")
    parser.add_argument("-H", "--height", type=int, default=600, help="height of resulting image")
    parser.add_argument("-w", "--width", type=int, default=800, help="width of resulting image")
    parser.add_argument("-R", "--radius_max", type=int, default=40, help="maximum of the radius of a circle")
    parser.add_argument("-r", "--radius_min", type=int, default=3, help="minimum of the radius of a circle")
    parser.add_argument("-a", "--max_attempts", type=int, default=2000,
                        help="maximum number of attempt to pack a circle")
    parser.add_argument("-v", "--verbose", action="store_const", const=True, default=False, help="verbosity")
    parser.add_argument("-s", "--save_file", type=str, default="", help="save file location")
    parser.add_argument("-o", "--output_graphml_file", type=str, default="", help="save file for graphml file")
    args = parser.parse_args()

    fname = args.csv_file
    graph = gr.Graph()
    graph.read_file(fname, delim=';')
    graph.read_colors(args.graphml_file)
    colors = graph.colors
    indices = cp.sort_nodes(graph)
    names = np.array(graph.nodeNames)[indices]
    circles = cp.generate_circles(graph, indices)
    img = cv.imread(args.image)
    img = cv.resize(img, (args.width, args.height), interpolation=cv.INTER_CUBIC)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    polygon = thresh
    save_file = args.save_file
    graphml_file = args.output_graphml_file
    new_img, used, unused, usedIdx, unusedIdx = cp.pack_polygon(polygon, circles, names, colors,
                                                                max_attempts=args.max_attempts, img_width=args.width,
                                                                img_height=args.height, radius_min=args.radius_min,
                                                                radius_max=args.radius_max,
                                                                verbose=args.verbose)
    figure = plt.figure(figsize=(args.width/100, args.height/100))
    ax = figure.add_axes([0, 0, 1, 1])
    ax.axis('off')
    plt.imshow(new_img, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])

    # Output image file if specified
    if save_file != "":
        plt.savefig(save_file)

    # Output graphml file if specified
    if graphml_file != "":
        G = nx.Graph()
        for _, circ in enumerate(used):
            G.add_node(names[circ[cp.CIRCLE_I]], x=float(circ[cp.CIRCLE_X]), y=float(args.height - circ[cp.CIRCLE_Y]),
                       r=int(colors[names[circ[cp.CIRCLE_I]]][0]), g=int(colors[names[circ[cp.CIRCLE_I]]][1]),
                       b=int(colors[names[circ[cp.CIRCLE_I]]][2]), size=float(circ[cp.CIRCLE_R]))
        for i, row in enumerate(graph.adjMatrix[indices]):
            for j, col in enumerate(row[indices]):
                if col > 0:
                    G.add_edge(names[i], names[j], weight=col)
        nx.write_graphml(G, "./got.graphml", named_key_ids=True)

    plt.show()
