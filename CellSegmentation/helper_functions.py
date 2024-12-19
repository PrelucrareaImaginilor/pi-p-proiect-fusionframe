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
    if pixel_accuracy <0.1:
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


