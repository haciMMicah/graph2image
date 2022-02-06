import numpy as np
import cv2 as cv
import graph as gr
from matplotlib import pyplot as plt

CIRCLE_Y = 0  # Circle attribute Y index
CIRCLE_X = 1  # Circle attribute X index
CIRCLE_R = 2  # Circle attribute R (Radius) index
CIRCLE_I = 3  # Circle attribute index in adj matrix

def sort_nodes(graph_obj):
    """
    sort_nodes returns a permutation array that corresponds to the
    indices of a graph's nodes sorted in descending order based on
    the outdegree of the nodes.

    :param graph_obj: Graph Object
    :return: Indices as a ndarray
    """
    # Sort the indices
    ind = np.argsort(graph_obj.nodeOutDegrees)
    ind = np.flip(ind)
    return ind


def generate_circles(graph_obj, indices, width=800, height=600):
    """
    Generates circles with random x,y coordinatesthat correspond to a graph object's nodes

    :param graph_obj: A Graph object
    :param indices: The indices of the nodes (should be in descended sorted order by outdegree)
    :param width: Width of image frame to generate the circles on
    :param height: Height of image fram to generate the circles on
    :return: ndarray that represents cirlce attributes [height, width, size, index] Nx4 where N = num circles
    """
    circle_attr = np.random.randint([0, 0, 0, 0], high=[height, width, 1, 1], size=(indices.shape[0], 4))
    circle_attr[:, 2] = graph_obj.nodeOutDegrees[indices]
    circle_attr[:, 3] = np.arange(indices.shape[0])
    return circle_attr


def point_inside_polygon(polygon, p, thresh_val=0):
    """
    Returns true if the point p lies inside the polygon (ndarray)
    Checks if point lies in a thresholded area

    :param polygon: ndarray that represents the thresholded image contour that we want to circle pack
    :param p: ndarray 1x2 representing a point in Euclidean N^2 space.
    :param thresh_val: The threshold value that we want to represent inclusion. i.e. if point (x, y)
           in the polygon is <= thresh_val then the point is in the polygon
    :return: True if point p in inside the polygon, False otherwise
    """
    p_x = p[1]
    p_y = p[0]
    if 0 <= p_x < polygon.shape[1] and 0 <= p_y < polygon.shape[0]:
        if polygon[p_y, p_x] <= thresh_val:
            return True
    return False


def circles_collide(circle1, circle2):
    """
    Returns True if  two given circles intersect

    :param circle1: ndarray, 1x4
    :param circle2: ndarray, 1x4
    :return: True if the circles intersect, False otherwise
    """
    c1_x = circle1[CIRCLE_X]
    c1_y = circle1[CIRCLE_Y]
    c1_r = circle1[CIRCLE_R]
    c2_x = circle2[CIRCLE_X]
    c2_y = circle2[CIRCLE_Y]
    c2_r = circle2[CIRCLE_R]

    dist = np.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    return dist <= c1_r + c2_r


def draw_circles(circles, names, colors, img_width=800, img_height=600):
    """
    Creates an image with a passed in array of circles drawn on it given label and color info
    :param circles: ndarray, Nx4 circle attributes
    :param names: Node labels to index int the color dictionary
    :param colors: Dictionary of node labels to color information
    :param img_width: int image height defaults to 800
    :param img_height: int image width defaults to 600
    :return: ndarray img_width X img_height sized Image with the drawn circles
    """
    img = np.zeros((img_height, img_width, 3), dtype='uint8')
    for idx, row in enumerate(circles[:]):
        index = circles[idx, CIRCLE_I]
        name = names[index]
        color = colors[name]
        thickness = -1
        center = (circles[idx, CIRCLE_X], circles[idx, CIRCLE_Y])
        radius = int(circles[idx, CIRCLE_R])
        img = cv.circle(img, center, radius, color, thickness)
    return img


def pack_polygon(polygon, circles, names, colors, img_width=800, img_height=600, max_attempts=2000, radius_min=3,
                 radius_max=40, verbose=False):
    """
    Returns an image matrix with the circles drawn to fill a polygon
    also returns the used/unused circles and used/unused indices of original circles array.
    Randomly places circles in polygon until they don't overlap.
    Derived from https://tylerxhobbs.com/essays/2016/a-randomized-approach-to-cicle-packing

    :param polygon: ndarray of binary thresholded contour image that we want to circle pack
    :param circles: ndarray of Nx4 circle attributes [height, width, size, index]
    :param names: ndarray of node labels
    :param colors: Dictionary of node labels to color information
    :param img_width: int width of the return image default 800
    :param img_height: int height of the return image default 600
    :param max_attempts: int maximum number of attempts to place a circle default 2000
    :param radius_min: int minimum radius of a circle to draw default 3, clips radius to min
    :param radius_max: int max radius of a circle to draw default 40, clips radius to max
    :param verbose: bool to print output or not default False
    :return: ndarray img, ndarray used circles, ndarray unused circles, ndarray usedIndices, ndarray unusedIndices
    """
    unused = []
    unusedIndices = []
    used = []
    usedIndices = []
    poly_max = polygon.shape
    poly_min = np.array([0, 0])
    for idx, circ in enumerate(circles[:]):
        placed_circle = False
        circ[CIRCLE_R] = int(circ[CIRCLE_R] * .4)
        if circ[CIRCLE_R] < radius_min:
            circ[CIRCLE_R] = radius_min
        if circ[CIRCLE_R] > radius_max:
            circ[CIRCLE_R] = radius_max
        for tries in range(max_attempts):
            # randomly place the polygon
            circ[CIRCLE_X] = np.random.randint(poly_min[1], poly_max[1])
            circ[CIRCLE_Y] = np.random.randint(poly_min[0], poly_max[0])
            # Check if the circle lies within polygon by going over 8 points around circle and checking each point
            is_inside = True
            for k in range(8):
                theta = np.pi * (k / 4.0)
                point_x = circ[CIRCLE_X] + int(circ[CIRCLE_R] * np.cos(theta))
                point_y = circ[CIRCLE_Y] + int(circ[CIRCLE_R] * np.sin(theta))
                if point_inside_polygon(polygon, np.array([point_y, point_x])):
                    pass
                else:
                    is_inside = False
                    break
            if is_inside:
                # Check collisions with other circles
                any_collisions = False
                for other in used:
                    if circles_collide(circ, other):
                        any_collisions = True
                        break
                if not any_collisions:
                    used.append(circ)
                    usedIndices.append(idx)
                    placed_circle = True
                    if verbose:
                        print("Placed node {}, Attributes: {}, Color: {}".format(names[idx], circ, colors[names[idx]]))
                    break
        if not placed_circle:
            unused.append(circ)
            unusedIndices.append(idx)
            if verbose:
                print("Did not place node {}, Attributes: {}".format(names[idx], circ))

    # Draw the circles
    used = np.array(used)
    img = draw_circles(used, names, colors, img_width, img_height)
    return img, used, unused, usedIndices, unusedIndices


if __name__ == "__main__":
    fname = '../resources/WitcherNetwork.csv'
    graph = gr.Graph()
    graph.read_file(fname, delim=';')
    graph.read_colors("../resources/WitcherNetwork.graphml")
    colors = graph.colors
    indices = sort_nodes(graph)
    names = np.array(graph.nodeNames)[indices]
    circles = generate_circles(graph, indices)
    img = cv.imread('../resources/witcherMedallion.jpg')
    img = cv.resize(img, (1000, 1000), interpolation=cv.INTER_CUBIC)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    polygon = thresh
    new_img, used, unused, usedIdx, unusedIdx = pack_polygon(polygon, circles, names, colors, max_attempts=500, img_width=1000,
                                                             img_height=1000, radius_min=5, radius_max=100, verbose=True)
    plt.subplot(121), plt.imshow(new_img, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])
    plt.show()
