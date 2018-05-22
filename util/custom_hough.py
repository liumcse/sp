import numpy as np
import cv2
from util.custom_canny import detect as canny_detect

from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line

def detect(grey, threshold=None):
    '''
    Detect lines using hough transform

    Parameters
    ----------
    grey: greyscale image after edge detection
    threshold: int
        Default is 0.85 * median of all votes.
    
    Returns
    -------
    strong_lines: ndarray
        [rho, theta, vote]
        Optimized lines.
    '''
    accumulator, thetas, rhos = hough_line(grey)
    
    # auto-threshold
    if threshold is None:
        acc_unique = np.unique(accumulator)
        acc_median = 0.85*int(np.round(np.median(acc_unique)))
        # print("median is " + str(acc_median))
        threshold = acc_median

    filtered = np.argwhere(accumulator > threshold)  # select lines whose vote > threshold
    lines = np.zeros([len(filtered), 3], dtype=int)

    for i in range(len(filtered)):
        rho, theta = filtered[i]
        vote = accumulator[rho, theta]
        lines[i] = rho, theta, vote

    n2 = 0
    strong_lines = np.zeros([len(lines), 3], dtype=float)  # A 3d-np-array. 10000 == up to 10000 lines (but decided by n2), 1 == simply for adapting to nd-array format, no real meaning, 2 == one for storing rho, one for storing theta
    for n1 in range(len(lines)):
        rho_idx, theta_idx, vote = lines[n1]
        rho = rhos[rho_idx]
        theta = thetas[theta_idx]
        if n1 == 0:
            strong_lines[n2] = rho, theta, vote
            n2 = n2 + 1
        else:
            closeness_rho = np.isclose(rho,strong_lines[0:n2,0], atol = 50)  # return an array of True/False
            closeness_theta = np.isclose(theta,strong_lines[0:n2,1], atol = np.pi/18)  # return an array of True/False
            closeness = np.all([closeness_rho,closeness_theta], axis=0)  # return an array of True/False
            if not any(closeness):  # if all false (no nearby line at all)
                strong_lines[n2] = rho, theta, vote
                n2 = n2 + 1
            else:  # if there is any True
                numLine = 0
                for i in range(len(closeness)):
                    if closeness[i] == True:
                        numLine = i
                        break
                _, _, this_vote = strong_lines[numLine]
                if vote > this_vote:
                    strong_lines[numLine] = rho, theta, vote
    return strong_lines[0:n2]

if __name__ == "__main__":
    # load image
    src = cv2.imread("../img/building.jpg")
    assert(src is not None)

    # detect edges
    edges = canny_detect(src)

    # detect lines
    lines = detect(edges)

    # name windows
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("lines", cv2.WINDOW_NORMAL)

    # draw lines
    line_output = src.copy()
    height, width, _ = src.shape
    diag_len = np.sqrt(height**2 + width**2)
    count = 0
    for line in lines:
        count = (count + 1) % 4
        rho,theta, _ = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + diag_len*(-b))
        y1 = int(y0 + diag_len*(a))
        x2 = int(x0 - diag_len*(-b))
        y2 = int(y0 - diag_len*(a))
        if count == 0:
            cv2.line(line_output,(x1,y1),(x2,y2),(0,0,255),1)
        elif count == 1:
            cv2.line(line_output,(x1,y1),(x2,y2),(0,255,0),1)
        elif count == 2:
            cv2.line(line_output,(x1,y1),(x2,y2),(128,128,128),1)
        else:
            cv2.line(line_output,(x1,y1),(x2,y2),(255,0,0),1)


    # display images
    cv2.imshow("original", src)
    cv2.imshow("edges", edges)
    cv2.imshow("lines", line_output)

    cv2.waitKey()
    cv2.destroyAllWindows()
    