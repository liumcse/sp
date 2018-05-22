import cv2
import numpy as np
import skimage.feature

def auto_threshold_otsu(image):
    '''
    This method finds threshold through Otsu thresholding
    About why it works: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf

    Parameters
    ----------
    image: grayscale image

    Returns
    -------
    lo: float
    hi: float
    '''
    # just a temp variable to fit in the method. Will not be used
    dst = np.zeros(image.shape)

    # Otsu thresholding
    hi, dst = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    lo = hi * 0.5

    return lo, hi


def detect(src):
    '''
    This method finds edges.

    Parameters
    ----------
    src: color image

    Returns
    -------
    edges: single-channel image of detected edges

    Notes
    -----
    Find optimized parameter for bilateral filter: http://ieeexplore.ieee.org.ezlibproxy1.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=7292752
    '''
    # convert into grayscale
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # smooth using bilateral filter
    # find optimized parameter for bilateral filter: http://ieeexplore.ieee.org.ezlibproxy1.ntu.edu.sg/stamp/stamp.jsp?tp=&arnumber=7292752
    blurred = cv2.bilateralFilter(gray, 17, 17, 17)
    # blurred = gray

    # auto threshold
    lo, hi = auto_threshold_otsu(gray)

    # edge detection
    edges = cv2.Canny(blurred, lo, hi, apertureSize=3)

    return edges


if __name__ == "__main__":
    # load image
    src = cv2.imread("../img/pyramid.jpg")
    assert(src is not None)

    # canny
    edges = detect(src)

    # name windows
    cv2.namedWindow("original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
    
    # display images
    cv2.imshow("original", src)
    cv2.imshow("edges", edges)
    
    cv2.waitKey()
    cv2.destroyAllWindows()