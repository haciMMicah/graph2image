import numpy as np
import cv2 as cv
import graph as gr
import circle_pack as cp
from matplotlib import pyplot as plt
import argparse


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
    new_img, used, unused, usedIdx, unusedIdx = cp.pack_polygon(polygon, circles, names, colors,
                                                                max_attempts=args.max_attempts, img_width=args.width,
                                                                img_height=args.height, radius_min=args.radius_min,
                                                                radius_max=args.radius_max,
                                                                verbose=args.verbose)
    plt.subplot(121), plt.imshow(new_img, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])
    plt.show()
