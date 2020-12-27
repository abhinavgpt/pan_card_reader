import numpy as np
import cv2
import os
import imutils
from tesserocr import PyTessBaseAPI, PSM
from PIL import Image
import re

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 4))

def cv_show(img, title="Image", is_debug=False):
    """
    Helper function to display an Image
    Inputs: title: The title of window in which image is to be displayed. 'Image' by default
            is_debug: Flag that dictates whether to actually show an Image.
                      Set to True to display image. False by default
    Returns: None. If is_debug is True, then an image will be shown in a separate window.
             Exit the window by pressing any key.
    """
    if is_debug:
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def sort_contours(cnts, method="left-to-right"):
    """
    Function to sort contours
    Inputs: cnts:  contours
            method: Which way the contours need to be sorted
                    Options: 'left-to-right', 'bottom-to-top',
                             'right-to-left', 'top-to-bottom'
    Returns: A tuple in the form of (sorted contours, bounding_boxes)
    """
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    return (cnts, boundingBoxes)

def extract_rois(img):
    """
    Function to extract ROIs from the Image
    Inputs: img: OpenCV image of PAN card
    Returns: A list of ROIs
    """
    # Information to be extracted from PAN card is in dark black color,
    # Locating the darkest regions in the image will give us the information required

    # Set this to True to see the different stages image goes through before OCR
    is_debug=False

    # Resizing the image in case it is too large
    height, width, channels = img.shape
    new_height = 600
    img = imutils.resize(img, height=new_height) # imutils.resize will conserve aspect ratio
    cv_show(img, title="Read file", is_debug=is_debug)

    orig = img.copy()
    # Converting to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show(img, title="Gray image", is_debug=is_debug)

    # Blurring the image to remove noise
    img = cv2.GaussianBlur(img, (3,3), 1)
    cv_show(img, title="Blurred image", is_debug=is_debug)

    # Contrast strecthing to make the blacks darker
    img = cv2.normalize(img, None, alpha=0, beta=1.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = np.clip(img, 0, 1)
    img = (255*img).astype(np.uint8)
    cv_show(img, title="Contrast stretched", is_debug=is_debug)

    # Applying the blackhat operator to convert the black regions into white regions
    # Note that after this operation we need to look at the brightest regions in the image
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)
    blackhatted = img.copy()
    cv_show(img, title="Blackhatted", is_debug=is_debug)

    # Again contrast stretching
    img = cv2.normalize(img, None, alpha=0, beta=1.8, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = np.clip(img, 0, 1)
    img = (255*img).astype(np.uint8)
    cv_show(img, title="Contrast stretched Again", is_debug=is_debug)

    # Merging pieces of information into groups
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, rectKernel)
    cv_show(img, "Closed", is_debug=is_debug)

    # Calculating the 95th percentile value to identify the brightest regions
    percentile = int(np.percentile(img, 95))
    # Sometimes 95th percentie may turn out to be a really high number
    percentile = min(240, percentile)

    # Thresholding the image using the percentile obtained
    img = cv2.threshold(img, percentile, 255, cv2.THRESH_BINARY)[1]
    cv_show(img, "Threshed", is_debug=is_debug)

    # find contours in the thresholded image and sort them from top to bottom
    # top-to-bottom is needed since in PAN card, name comes before father's name
    cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")

    rois = []
    # Loop over the contours and select the ones that satisfy our criteria
    for c in cnts:

        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Calculating aspect ratio and coverage width of the contour
        # These parameters can be used to identify rectangular contours
        ar = w / float(h)
        crWidth = w / float(orig.shape[1])

        if ar > 5 and crWidth > 0.10 and crWidth < 0.6:

            # Adding padding to the contour since Tesseract will need it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.02)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            # extract the ROI from the image
            roi = orig[y:y + h, x:x + w].copy()
            cv_show(roi, "Region of interest", is_debug=is_debug)
            rois.append(roi)
        #                 roi = cv2.filter2D(roi, -1, sharpen_kernel)
        #                 cv_show(roi, "Sharpened", is_debug=is_debug)
    return rois

def ocr_roi(rois):
    """
    Function to do the OCR
    Inputs: rois: A list of Images
    Returns: A List of strings. One for each incoming Image
    """
    # Initializing PyTessBaseApi again and again adds an unnecessary overhead.
    # So initializing it just once
    # PSM.SINGLE_LINE degrades performace
    extracted_text = []
    with PyTessBaseAPI(path=os.getcwd()) as api:
        for roi in rois:
            # Converting the OpenCV image to PIL image for tesseract
            api.SetImage(Image.fromarray(roi))
            extracted_text.append(api.GetUTF8Text())
    return extracted_text

def extract_fields(extracted_info):
    """
    Function to extract required information from the text parsed by OCR
    Inputs: extracted_info: A list of strings
    Return: A dictionary containing the required Information
    """

    # This is pattern for matching lines that are not of use to us
    reject_pat = r"\bGOVT\b|\bOF\b|\bINDIA\b|\bNumber\b|\bINCOME\b|\bTAX\b|\bDEPARTMENT\b"

    # Identifying names. Utilizing the fact that are going to be in ALL CAPS
    name_pat = r"^[A-Z ]+$"

    # Pattern for PAN card
    pan_pat = r"^[A-Z]+[0-9]+[A-Z]+$"

    pan, date = None, None
    names = []

    # Looping over all the extracted strings to find the strings that match patterns above
    for elem in extracted_info:
        elem = elem.strip()

        if re.search(reject_pat, elem):
            continue

        if re.search(name_pat, elem):
            if len(elem.split()) <= 4:
                names.append(elem)
        elif re.search(pan_pat, elem):
            pan = elem
        elif elem.count(r"/") >= 1:
            date = elem

    name, fname = None, None
    if len(names) >= 2:
        name, fname = names[:2]
    elif len(names) == 1:
        name = names[0]
    return {'name':name, 'fname':fname, 'pan':pan, 'date':date}

def extract_info(img):
    """
    Takes an OpenCV image of PAN card as input and returns extracted information in a dictionary
    """
    rois = extract_rois(img)
    ocr_info = ocr_roi(rois)
    extracted_info = extract_fields(ocr_info)
    return extracted_info
