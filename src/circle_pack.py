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


# Given 3 collinear points, checks if point q lies on segment pr. Derived from
# https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
def on_segment(p, q, r):
    q_x = q[1]
    q_y = q[0]
    p_x = p[1]
    p_y = p[0]
    r_x = r[1]
    r_y = r[0]
    if max(p_x, r_x) >= q_x >= min(p_x, r_x) and max(p_y, r_y) >= q_y >= min(p_y, r_y):
        return True
    else:
        return False


# Given point triplet (p, q, r) use cross product to find orientation of the triplet
# derived from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
# 0 -> p, q, and r are collinear
# 1 -> clockwise
# 2 -> couter clockwise
def orientation(p, q, r):
    q_x = q[1]
    q_y = q[0]
    p_x = p[1]
    p_y = p[0]
    r_x = r[1]
    r_y = r[0]
    val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y)
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

# Returns true if line segment p1q1 and p2q2 intersect
# derived from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1, and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1, and p2 are collinear and q1 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2, and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2, and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


# Returns true if the point p lies inside the polygon (ndarray)
# derived from https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
def point_inside_polygon(polygon, p):
    n = polygon.shape[0]
    p_x = p[1]
    p_y = p[0]

    # Polygon must have at least 3 vertices
    if n < 3:
        return False

    # create some large value to be our infinite value TODO: (revisit this later)
    inf = 500000

    # Create a point for line segment from p to inf
    line = np.array([p_y, inf])

    count = 0
    i = 0
    is_done = False
    # count intersection of the above line with sides of the polygon
    while not is_done:
        next = (i + 1) % n
        # Check if the line segment from p to line intersects with the line segment from polygon[i] to polygon[next]
        if do_intersect(polygon[i, :], polygon[next, :], p, line):
            # If the point p is collinear with line segment i->next, then check if it lies on segment.
            # if it lies, return true, otherwise false
            if orientation(polygon[i], p, polygon[next]) == 0:
                return on_segment(polygon[i], p, polygon[next])
            count += 1
        i = next
        is_done = i == 0
    return count % 2 == 1

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


def draw_circles(circles, polygon=None, img_width=800, img_height=600):
    img = np.zeros((img_height, img_width, 3), dtype='uint8')
    colors = np.random.randint([0, 0, 0], high=[255, 255, 255], size=(circles.shape[0], 3))
    for idx, row in enumerate(circles[:]):
        color = (int(colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2]))
        thickness = -1
        center = (circles[idx, 1], circles[idx, 0])
        radius = int(circles[idx, 2])
        img = cv.circle(img, center, radius, color, thickness)
        if polygon is not None:
            img = cv.polylines(img, [polygon], True, (255, 255, 255))
    return img

# Returns an image matrix with the circles drawn to fill a polygon
# also returns the unused circles. Randomly places circles in polygon
# until they don't overlap. derived from https://tylerxhobbs.com/essays/2016/a-randomized-approach-to-cicle-packing
def pack_polygon(polygon, circles, img_width=800, img_height=600, max_attempts=2000, radius_min=3, radius_max=40):
    unused = []
    used = []
    poly_max = polygon.max(axis=0)
    poly_min = polygon.min(axis=0)
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
    img = draw_circles(used, polygon, img_width, img_height)
    return img, unused


if __name__ == "__main__":
    fname = '../resources/WitcherNetwork.csv'
    graph = gr.Graph()
    graph.read_file(fname, delim=';')
    indices = sort_nodes(graph)
    circles = generate_circles(graph, indices)
    polygon = np.array([[0, 0], [0, 400], [400, 400], [400, 0]], dtype='int')
    new_img, unused = pack_polygon(polygon, circles, max_attempts=20)
    print(unused)
    plt.subplot(121), plt.imshow(new_img, cmap='gray')
    plt.title('circles'), plt.xticks([]), plt.yticks([])
    plt.show()
