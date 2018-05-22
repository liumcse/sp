from util.custom_canny import detect as canny_detect
from util.custom_hough import detect as hough_detect

import cv2
import numpy as np

def _normalize(theta):
    while theta > np.pi:
        theta = theta - 2*np.pi
    while theta < -np.pi:
        theta = theta + 2*np.pi
    return theta

def first_pass(src):
    '''
    First pass

    Parameters
    ----------
    src: BGR image

    Returns:
    edges: greyscale image with detected edges
    edge_point: list
        a list of points (x, y) represents location of edge pixels
    '''
    assert(src is not None)

    edges = canny_detect(src)
    height, width = edges.shape
    edge_point = []    
    for i in range(height):
        for j in range(width):
            if edges[i, j] != 0:  # is an edge point
                edge_point.append((j, i))


    print("First pass done.")

    return edges, edge_point

def second_pass(edges, edge_points, threshold=None, output_image=None):
    '''
    Second pass

    Parameters
    ----------
    edges: greyscale image with detected edges
    edge_points: list
        a list of points (x, y) represents location of edge pixels
    threshold: int
        Default is 0.85 * median of all votes.

    Returns:
    --------
    intersect_points_with_direction: dictionary
        {(x, y) : theta (in radius)}
    '''

    assert(edges is not None)
    # another assert here

    # variables
    count = 0
    height, width = edges.shape
    lines =  hough_detect(edges, threshold=threshold)
    total_count = str(len(lines))
    intersect_points_with_direction = dict()

    for l1 in lines:
        count = count + 1
        print("Processing " + str(count) + "/" + total_count + " of second pass.")
        rho1, theta1, _ = l1
        temp_intersect_point = set()
        for l2 in lines:
            rho2, theta2, _ = l2
            if theta1 == theta2:  # parallel lines, skip
                continue  
            # solve the linear system to get intersection point
            a = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array(
                [rho1, rho2]
            )
            x, y = np.linalg.solve(a, b)
            if x > width or x < 0 or y > height or y < 0:  # if out of bound, don't add
                continue
            check = 0
            # check if it is a standalone point
            for i in range(-5, 5):
                for j in range(-5 ,5):
                    x1 = int(np.round(x + i))
                    y1 = int(np.round(y + j))
                    if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
                        check = check + 0
                    else:
                        if (x1, y1) in edge_points:
                            check = check + 1
            # print("Before check " + str(x) + ", " + str(y))
            if check == 0:
                continue
            # print("After check")
            if (x, y) not in intersect_points_with_direction.keys():
                intersect_points_with_direction[(x, y)] = set()
            temp_intersect_point.add((x, y))
            # print("Added")
        visited = list()
        # print(len(temp_intersect_point))
        # find direction at each intersection point in this line
        for point1 in temp_intersect_point:
            visited.append(point1)
            min_point = None
            min_distance = float('inf')
            x1, y1 = point1
            for point2 in temp_intersect_point:
                if point2 in visited:
                    continue
                x2, y2 = point2
                distance = (x1 - x2)**2 + (y1 - y2)**2
                if distance < min_distance and distance >= 10:  # distance >= 10 to skip overlapping points
                    min_distance = distance
                    min_point = (x2, y2)
            if min_point is not None:
                x2, _ = min_point
                _x2 = x2 - x1
                _x1 = 1*np.cos(theta1 + np.pi/2)
                if _x1*_x2 < 0:
                    _theta = _normalize(theta1 - np.pi)
                else:
                    _theta = theta1
                intersect_points_with_direction[point1].add(_theta)
                intersect_points_with_direction[min_point].add(_normalize(_theta - np.pi))

    assert(len(intersect_points_with_direction.keys()) > 0)
    
    # round to nearest integer. lose precision. 
    temp = list(intersect_points_with_direction.keys())
    for point in temp:
        x, y = int(np.round(point[0])), int(np.round(point[1]))
        intersect_points_with_direction[(x, y)] = intersect_points_with_direction[(point[0], point[1])]
        del intersect_points_with_direction[(point[0], point[1])]

    if output_image is not None:
        count = 0
        for direction_vector in intersect_points_with_direction.keys():
            count = (count + 1) % 3
            p = direction_vector
            theta = intersect_points_with_direction[p]
            x1, y1 = p
            assert(type(theta) == set)
            for _theta in theta:
                arrow_length = 30
                a = np.cos(_normalize(np.pi/2 + _theta))
                b = np.sin(_normalize(np.pi/2 + _theta))
                x3 = int(np.round(x1 + arrow_length*a))
                y3 = int(np.round(y1 + arrow_length*b))
                if count == 0:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (0, 0, 255), thickness=2, tipLength=0.2)
                elif count == 1:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (0, 255, 0), thickness=2, tipLength=0.4)
                elif count == 2:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (255, 0, 0), thickness=2, tipLength=0.6)  

    assert(len(intersect_points_with_direction.keys()) > 0)
    print("Second pass done.")
    return intersect_points_with_direction

def third_pass(edges, intersect_points_with_direction, output_image=None):
    '''
    Third pass

    Parameters
    ----------
    edges: greyscale image with detected edges
    intersect_points_with_direction: dictionary
        {(x, y) : theta (in radius)}
    
    Returns
    -------
    intersect_points_with_direction: dictionary
        {(x, y) : theta (in radius)}
        With overlapping points removed
    '''
    assert(edges is not None)
    # another assert here

    intersect_points = intersect_points_with_direction.keys()
    intersect_count = 0
    to_delete = set()
    to_insert = dict()
    len_intersect_points_with_direction = len(intersect_points_with_direction)
    for point1 in intersect_points:
        overlap_points = set()
        overlap_points.add(point1)
        intersect_count = intersect_count + 1     
        print("Processing " + str(intersect_count) + " out of " + str(len_intersect_points_with_direction) + " of third pass")        
        for point2 in intersect_points:
            if point1 != point2:
                x1, y1 = point1
                x2, y2 = point2
                dist = np.sqrt((y1-y2)**2 + (x1-x2)**2)
                if dist <= 15:
                    overlap_points.add(point2)
        if len(overlap_points) > 1:
            aggregated_directions = set()
            for point in overlap_points:
                for direction in intersect_points_with_direction[point]:
                    aggregated_directions.add(direction)
                to_delete.add(point)
            x_sum = 0
            y_sum = 0
            for point in overlap_points:
                x_sum += point[0]
                y_sum += point[1]
            computed_x = int(np.round(x_sum / len(overlap_points)))
            computed_y = int(np.round(y_sum / len(overlap_points)))
            computed_point = computed_x, computed_y
            to_insert[computed_point] = aggregated_directions

    for point in to_delete:
        del intersect_points_with_direction[point]
    for point_direction in to_insert.keys():
        intersect_points_with_direction[point_direction] = to_insert[point_direction]

    if output_image is not None:
        count = 0
        for direction_vector in intersect_points_with_direction.keys():
            count = (count + 1) % 3
            p = direction_vector
            theta = intersect_points_with_direction[p]
            x1, y1 = p
            assert(type(theta) == set)
            for _theta in theta:
                arrow_length = 30
                a = np.cos(_normalize(np.pi/2 + _theta))
                b = np.sin(_normalize(np.pi/2 + _theta))
                x3 = int(np.round(x1 + arrow_length*a))
                y3 = int(np.round(y1 + arrow_length*b))
                if count == 0:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (0, 0, 255), thickness=2, tipLength=0.2)
                elif count == 1:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (0, 255, 0), thickness=2, tipLength=0.4)
                elif count == 2:
                    cv2.arrowedLine(output_image, (x1, y1), (x3, y3), (255, 0, 0), thickness=2, tipLength=0.6) 

    print("Third pass done.")
    return intersect_points_with_direction

def fourth_pass(output_image, intersect_points_with_direction, epsilon=0.05):
    '''
    Fourth pass

    Parameters
    ----------
    output_image:
    intersect_points_with_direction

    Returns
    -------
    None
    '''
    assert(intersect_points_with_direction is not None)
    # another assert here

    len_intersect_points_with_direction = len(intersect_points_with_direction)
    intersect_count = 0
    visited = list()
    for p1 in intersect_points_with_direction.keys():    
        intersect_count = intersect_count + 1 
        print("Processing " + str(intersect_count) + " out of " + str(len_intersect_points_with_direction) + " of fourth pass")                
        # if p1 in checked:
        #     continue
        for p2 in intersect_points_with_direction.keys():
            to_del_1 = set()  
            to_del_2 = set()    
            if p1 == p2:  # if they are the same vector, skip
                continue
            theta1 = intersect_points_with_direction[p1]
            theta2 = intersect_points_with_direction[p2]
            assert(type(theta1) == set)
            assert(type(theta2) == set)            
            if p1 == p2:  # if they are based on the same point... skip
                continue
            for _theta1 in theta1:
                for _theta2 in theta2:
                    normalized_theta1 = _normalize(np.pi/2 + _theta1)
                    normalized_theta2 = _normalize(np.pi/2 + _theta2)
                    if p2[0] - p1[0] == 0:
                        theta_in_between = np.pi/2
                    else:
                        theta_in_between = _normalize(np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0])))
                    if np.abs(normalized_theta1 - normalized_theta2 - np.pi) > epsilon \
                        and np.abs(normalized_theta1 - normalized_theta2 + np.pi) > epsilon:
                        continue
                    if np.abs(normalized_theta1 - theta_in_between - np.pi) > epsilon \
                        and np.abs(normalized_theta1 - theta_in_between + np.pi) > epsilon:
                        continue
                    if (p1, _theta1) in visited or (p2, _theta2) in visited:
                        continue
                    cv2.line(output_image, p1, p2, (0, 0, 0), 5)    
                    visited.append((p1, _theta1))
                    visited.append((p2, _theta2))  
    print("Fourth pass done.")
    print("All done.")

def execute(src, threshold=None):
    assert(src is not None)
    output_image = 255 * np.ones(src.shape)
    # output_image = src.copy()

    # first pass
    edges, edge_points = first_pass(src)
    # second pass
    second_pass_output = src.copy()
    intersect_points_with_direction = second_pass(edges, edge_points, threshold, output_image=second_pass_output)
    # third pass
    third_pass_output = src.copy()
    intersect_points_with_direction = third_pass(edges, intersect_points_with_direction, third_pass_output)
    # fourth pass
    fourth_pass(output_image, intersect_points_with_direction)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("First pass (Edges)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Second pass", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Third pass", cv2.WINDOW_NORMAL)    
    cv2.namedWindow("Fourth pass (Output)", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", src)
    cv2.imshow("First pass (Edges)", edges)
    cv2.imshow("Second pass", second_pass_output)
    cv2.imshow("Third pass", third_pass_output)    
    cv2.imshow("Fourth pass (Output)", output_image)
    cv2.waitKey()

if __name__ == "__main__":
    src = cv2.imread("../img/webgl.png")
    execute(src)