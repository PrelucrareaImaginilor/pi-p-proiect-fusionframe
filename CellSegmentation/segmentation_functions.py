import cv2
import numpy as np
import helper_functions as hp

def custom_segmentation(img, gray_img, annotated_mask):
    print("Custom segmentation algoritm")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(gray_img)
    # cv2.imshow("equalized", equalized_img)

    # noise reduction
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Blur", img)
    blurred_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    cv2.imshow("Gaussian blur", blurred_img)

    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Binary Image", binary_img)
    cv2.imshow("Annotated Mask", annotated_mask)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)  # Green contours for cells

    f_measure = hp.statistics(annotated_mask, binary_img)
    return img, f_measure


def improved_watershed(input_img, annotated_mask):
    # Step 1: Noise reduction with Gaussian blur
    blurred_img = cv2.GaussianBlur(input_img, (5, 5), 0)

    # Step 2: Otsu's Thresholding to create binary image of cells
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Remove noise with morphological opening (adjust the kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=3)

    # Step 4: Distance transform to find sure foreground (cells)
    dist_transform = cv2.distanceTransform(opened_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)  # Adjust threshold
    sure_fg = np.uint8(sure_fg)

    # Step 5: Detect sure background (dilating binary image with larger kernel)
    sure_bg = cv2.dilate(opened_img, kernel, iterations=5)  # Increase iterations for clearer background detection
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Step 6: Marker labelling (mark foreground with 1, background with 0)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background is marked as 1, cells as other labels
    markers[unknown == 255] = 0  # Unknown regions marked as 0

    # Step 7: Apply Watershed to segment the cells
    img_for_watershed = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_for_watershed, markers)

    # Step 8: Mark the watershed boundaries (now with more clarity)
    img_for_watershed[markers == -1] = [0, 0, 255]  # Use red for boundaries

    # Step 9: Optionally, apply closing to clean up small noise
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing_img = cv2.morphologyEx(img_for_watershed, cv2.MORPH_CLOSE, closing_kernel)
    # Step 10: Return results
    # f_measure = statistics(annotated_mask, binary_img)
    return closing_img, binary_img, markers

