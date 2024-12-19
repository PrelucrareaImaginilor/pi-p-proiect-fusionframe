import cv2
import numpy as np


def statistics(annotated_mask, binary_img):
    white_mask = (annotated_mask == 255)
    black_mask = (annotated_mask == 0)

    white_img = (binary_img == 255)
    black_img = (binary_img == 0)

    # pixeli detectati din obiect corect
    true_pozitive = np.sum(white_mask & white_img)
    # pixeli din background detectati corect
    true_negative = np.sum(black_img & black_mask)
    # pixeli detectati din obiect dar nu sunt
    false_positive = np.sum(white_img & black_mask)
    # pixeli detectati din background dar sunt din obiect
    false_negative = np.sum(black_img & white_mask)

    pixel_accuracy = (true_pozitive + true_negative) / (true_pozitive + true_negative + false_positive + false_negative)
    if pixel_accuracy < 5:
        binary_img = 255 - binary_img
        white_mask = (annotated_mask == 255)
        black_mask = (annotated_mask == 0)

        white_img = (binary_img == 255)
        black_img = (binary_img == 0)

        # pixeli detectati din obiect corect
        true_pozitive = np.sum(white_mask & white_img)
        # pixeli din background detectati corect
        true_negative = np.sum(black_img & black_mask)
        # pixeli detectati din obiect dar nu sunt
        false_positive = np.sum(white_img & black_mask)
        # pixeli detectati din background dar sunt din obiect
        false_negative = np.sum(black_img & white_mask)

        pixel_accuracy = (true_pozitive + true_negative) / (
                true_pozitive + true_negative + false_positive + false_negative)

    print(f"Pixel accuracy is {pixel_accuracy}%")
    # Jaccard Index
    IoU = true_pozitive / (true_pozitive + false_negative + false_positive)
    print(f"Area of overlap/Area of Union is {IoU}%")

    # a metric of exactness or quality
    precision = true_pozitive / (true_pozitive + false_positive)
    print(f"Precision is {precision}%")

    # a metric of completeness or quantity
    recall = true_pozitive / (true_pozitive + false_negative)
    print(f"Recall is {recall}%")
    # precision and recall should be high, but they have an inverse relationship

    # dice similarity coefficient
    f_measure = (2 * recall * precision) / (recall + precision)
    print(f"f_measure is {f_measure}")
    # cu atat e mai mare f_measure cu atat e mai bun algoritmul
    return f_measure


def custom_segmentation(img, gray_img, annotated_mask):
    print("Custom segmentation algoritm")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(gray_img)
    # cv2.imshow("equalized", equalized_img)

    # noise reduction
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Blur", img)
    blurred_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    # cv2.imshow("Gaussian blur", blurred_img)

    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Binary Image", binary_img)
    # cv2.imshow("Annotated Mask", annotated_mask)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)  # Green contours for cells

    f_measure = statistics(annotated_mask, binary_img)
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


def main():
    i = 4
    img = cv2.imread(rf'./CompositeMasks/Images/annotatedImages/{i}/images/1.png')
    annotated_mask = cv2.imread(rf'./CompositeMasks/AnnotatedMasks/results_new_mask{i}.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray img", gray_img)
    contoured_img1, f_measure1 = custom_segmentation(img, gray_img, annotated_mask)

    # Run the improved watershed function
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    segmented_img, binary_img, markers = improved_watershed(enhanced_img, annotated_mask)
    cv2.imshow('Custom segmented', contoured_img1)
    cv2.imshow('WaterShed segmented', segmented_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
