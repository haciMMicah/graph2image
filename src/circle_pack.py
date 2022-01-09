import cv2
import numpy as np
import cv2 as cv
import graph as gr
from matplotlib import pyplot as plt


def sort_nodes(graph_obj):
    # Sort the indices
    ind = np.argsort(graph_obj.nodeOutDegrees)
    ind = np.flip(ind)
    return ind


def generate_circles(graph_obj, indices, width=800, height=600):
    circle_attr = np.random.randint([0, 0, 0], high=[height, width, 1], size=(indices.shape[0], 3))
    circle_attr[:, 2] = graph_obj.nodeOutDegrees[indices]
    return circle_attr


# Returns true if the point p lies inside the polygon (ndarray)
# Checks if point lies in a thresholded area
def point_inside_polygon(polygon, p, thresh_val=0):
    p_x = p[1]
    p_y = p[0]
    if 0 <= p_x < polygon.shape[1] and 0 <= p_y < polygon.shape[0]:
        if polygon[p_y, p_x] <= thresh_val:
            return True
    return False

# Returns true if two given circles intersect
def circles_collide(circle1, circle2):
    c1_x = circle1[1]
    c1_y = circle1[0]
    c1_r = circle1[2]
    c2_x = circle2[1]
    c2_y = circle2[0]
    c2_r = circle2[2]

    dist = np.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
    return dist <= c1_r + c2_r


def draw_circles(circles, img_width=800, img_height=600):
    img = np.zeros((img_height, img_width, 3), dtype='uint8')
    colors = np.random.randint([0, 0, 0], high=[255, 255, 255], size=(circles.shape[0], 3))
    for idx, row in enumerate(circles[:]):
        color = (int(colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2]))
        thickness = -1
        center = (circles[idx, 1], circles[idx, 0])
        radius = int(circles[idx, 2])
        img = cv.circle(img, center, radius, color, thickness)
    return img


# Returns an image matrix with the circles drawn to fill a polygon
# also returns the unused circles. Randomly places circles in polygon
# until they don't overlap. Derived from https://tylerxhobbs.com/essays/2016/a-randomized-approach-to-cicle-packing
def pack_polygon(polygon, circles, img_width=800, img_height=600, max_attempts=2000, radius_min=3, radius_max=40):
    unused = []
    used = []
    poly_max = polygon.shape
    poly_min = np.array([0, 0])
    for idx, circ in enumerate(circles[:]):
        placed_circle = False
        circ[2] = int(circ[2] * .4)
        if circ[2] < radius_min:
            circ[2] = radius_min
        if circ[2] > radius_max:
            circ[2] = radius_max
        for tries in range(max_attempts):
            # randomly place the polygon
            circ[1] = np.random.randint(poly_min[1], poly_max[1])
            circ[0] = np.random.randint(poly_min[0], poly_max[0])
            # Check if the circle lies within polygon by going over 8 points around circle and checking each point
            is_inside = True
            for k in range(8):
                theta = np.pi * (k / 4.0)
                point_x = circ[1] + int(circ[2] * np.cos(theta))
                point_y = circ[0] + int(circ[2] * np.sin(theta))
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
                    placed_circle = True
                    print("Placed circle {} at idx {}".format(circ, idx))
                    break
        if not placed_circle:
            unused.append(circ)
            print("Did not place circle {} at idx {}".format(circ, idx))

    # Draw the circles
    used = np.array(used)
    img = draw_circles(used, img_width, img_height)
    return img, unused


if __name__ == "__main__":
    fname = '../resources/WitcherNetwork.csv'
    graph = gr.Graph()
    graph.read_file(fname, delim=';')
    indices = sort_nodes(graph)
    circles = generate_circles(graph, indices)
    img = cv.imread('../resources/witcherMedallion.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    polygon = thresh
    new_img, unused = pack_polygon(polygon, circles, max_attempts=1000, img_width=1000,
                                   img_height=1000, radius_min=3, radius_max=50)
    plt.subplot(121), plt.imshow(new_img, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])
    plt.show()
