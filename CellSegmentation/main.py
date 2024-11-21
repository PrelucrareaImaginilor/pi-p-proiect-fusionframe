import cv2
import numpy as np


def main():
    # img = cv2.imread(r 'images/BloodImage_00296_jpg.rf.d1e417e0e79924d42be3fd98f8eba369.jpg')
    # img = cv2.imread(r 'images/BloodImage_00259_jpg.rf.66036dfe9522476d28771c04d40ee6b3.jpg')
    # img = cv2.imread(r'./images/cell1.png')
    img = cv2.imread(r'./images/cell_images/1.png')
    annotatedMask = cv2.imread(r'./images/masks/new_mask1.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("original", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray img", gray_img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(gray_img)
    # cv2.imshow("equalized", equalized_img)

    # noise reduction
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Blur", img)
    blurred_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    # cv2.imshow("Gaussian blur", blurred_img)

    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Binary Image", binary_img)
    cv2.imshow("Annotated Mask", annotatedMask)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)  # Green contours for cells

    # cv2.imshow('Segmented Cells', img)

    white_mask = (annotatedMask == 255)
    black_mask = (annotatedMask == 0)

    white_img = (binary_img == 255)
    black_img = (binary_img == 0)

    # pixeli detectati din obiect corect
    TP = np.sum(white_mask & white_img)
    # pixeli din background detectati corect
    TN = np.sum(black_img & black_mask)
    # pixeli detectati din obiect dar nu sunt
    FP = np.sum(white_img & black_mask)
    # pixeli detectati din background dar sunt din obiect
    FN = np.sum(black_img & white_mask)

    pixelAccuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Pixel accuracy is {pixelAccuracy}%")
    # Jaccard Index
    IoU = TP / (TP + FN + FP)
    print(f"Area of overlap/Area of Union is {IoU}%")

    # a metric of exactness or quality
    precision = TP / (TP + FP)
    print(f"Precision is {precision}%")

    # a metric of completeness or quantity
    recall = TP / (TP + FN)
    print(f"Recall is {recall}%")
    # precision and recall should be high, but they have an inverse relationship

    # dice similarity coefficient
    F_measure = (2 * recall * precision) / (recall + precision)
    print(f"F_measure is {F_measure}")
    # cu atat e mai mare F_measure cu atat e mai bun algoritmul

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
