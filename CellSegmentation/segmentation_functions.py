import cv2
import numpy as np
import helper_functions as hp
# from sklearn.cluster import KMeans

def custom_segmentation(img, gray_img, annotated_mask):
    print("Custom segmentation algoritm")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(gray_img)
    # cv2.imshow("equalized", equalized_img) # forteaza o histograma pe care dorim sa o obtinem peste imagine

    # noise reduction
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow("Blur", img)
    blurred_img = cv2.GaussianBlur(equalized_img, (5, 5), 0)
    #cv2.imshow("Gaussian blur", blurred_img)

    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow("Binary Image", binary_img)
    #cv2.imshow("Annotated Mask", annotated_mask)

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
#
# def clustering_segmentation(img, annotated_mask, num_clusters=3):
#     print("Clustering segmentation using K-Means")
#
#     pixel_values = img.reshape((-1, 3))
#     pixel_values = np.float32(pixel_values)
#
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     kmeans.fit(pixel_values)
#     labels = kmeans.labels_
#
#     segmented_image = labels.reshape((img.shape[:2]))
#
#     binary_img = (segmented_image == 1).astype(np.uint8) * 255
#     binary_img = cv2.medianBlur(binary_img, 5)
#
#     segmented_display = np.zeros_like(img)
#     for cluster in range(num_clusters):
#         mask = (segmented_image == cluster)
#         segmented_display[mask] = np.random.randint(0, 255, size=3)
#
#     f_measure = hp.statistics(annotated_mask, binary_img)
#
#     return segmented_display, binary_img, f_measure


def region_splitting_and_merging(img, threshold=15, min_size=20):
    h, w = img.shape[:2]
    segmented_img = np.zeros_like(img, dtype=np.uint8)
    regions = [(0, 0, w, h)]
    def split_region(x, y, width, height):
        region = img[y:y + height, x:x + width]
        std_dev = np.std(region)
        if std_dev > threshold:
            return [
                (x, y, width // 2, height // 2),
                (x + width // 2, y, width // 2, height // 2),
                (x, y + height // 2, width // 2, height // 2),
                (x + width // 2, y + height // 2, width // 2, height // 2)
            ]
        return [(x, y, width, height)]

    def merge_region(x, y, width, height):
        region = img[y:y + height, x:x + width]
        mean_val = np.mean(region)
        segmented_img[y:y + height, x:x + width] = mean_val if mean_val > 128 else 0

    while regions:
        x, y, width, height = regions.pop(0)
        if width > 1 and height > 1:
            sub_regions = split_region(x, y, width, height)
            if len(sub_regions) > 1:
                regions.extend(sub_regions)
            else:
                merge_region(x, y, width, height)

    kernel = np.ones((3, 3), np.uint8)
    final_segmented_img = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, kernel)

    return final_segmented_img

def region_growing(image, annotated_mask, threshold=10):
    height, width = image.shape
    segmented_img = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.bool_)

    _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels = cv2.connectedComponents(binary_img)

    for label in range(1, num_labels):
        seed_points = np.argwhere(labels == label)
        stack = [(pt[1], pt[0]) for pt in seed_points]

        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            intensity = image[y, x]

            if abs(intensity - image[seed_points[0][0], seed_points[0][1]]) <= threshold:
                segmented_img[y, x] = 255

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        stack.append((nx, ny))

    f_measure = hp.statistics(annotated_mask, segmented_img)
    return segmented_img, f_measure



def graph_cut_segmentation(img, annotated_mask):
    print("Applying Graph Cut Segmentation...")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)

    _, initial_mask = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(initial_mask, kernel, iterations=2)

    grabcut_mask = np.where(dilated_mask == 255, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, grabcut_mask, None, bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK)

    refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)

    f_measure = hp.statistics(annotated_mask, refined_mask)

    segmented_img = cv2.bitwise_and(img, img, mask=refined_mask)
    return segmented_img, refined_mask, f_measure
