import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

max_accuracy=0.0
max_IoU=0.0
max_precision=0.0
max_recall=0.0
max_f_measure=0.0

max_accuracy_caller=0.0
max_IoU_caller=0.0
max_precision_caller=0.0
max_recall_caller=0.0
max_f_measure_caller=0.0

nr_gaussian_and_otsu_accuracies=0
nr_watershed_accuracies=0
nr_clustering_accuracies=0
nr_region_growing_accuracies=0
nr_region_splitting_accuracies=0

nr_gaussian_and_otsu_f_measures=0
nr_watershed_f_measures=0
nr_clustering_f_measures=0
nr_region_growing_f_measures=0
nr_region_splitting_f_measures=0

nr_gaussian_and_otsu_IoUs=0
nr_watershed_IoUs=0
nr_clustering_IoUs=0
nr_region_growing_IoUs=0
nr_region_splitting_IoUs=0

nr_gaussian_and_otsu_precisions=0
nr_watershed_precisions=0
nr_clustering_precisions=0
nr_region_growing_precisions=0
nr_region_splitting_precisions=0

nr_gaussian_and_otsu_recalls=0
nr_watershed_recalls=0
nr_clustering_recalls=0
nr_region_growing_recalls=0
nr_region_splitting_recalls=0

global_pixel_accuracy = 0.0
global_IoU = 0.0
global_precision = 0.0
global_recall = 0.0
global_f_measure = 0.0

current_image = None
current_annotated_mask = None

current_segmented_image_gaussian_and_otsu = None
current_segmented_mask_gaussian_and_otsu = None

current_segmented_image_watershed = None
current_segmented_mask_watershed = None

current_segmented_image_clustering = None
current_segmented_mask_clustering = None

current_segmented_image_region_growing = None
current_segmented_mask_region_growing = None

current_segmented_image_region_splitting=None
current_segmented_mask_region_splitting=None


gaussian_pixel_accuracy = 0.0
gaussian_IoU = 0.0
gaussian_precision = 0.0
gaussian_recall = 0.0
gaussian_f_measure = 0.0
gaussian_count = 0

watershed_pixel_accuracy = 0.0
watershed_IoU = 0.0
watershed_precision = 0.0
watershed_recall = 0.0
watershed_f_measure = 0.0
watershed_count = 0

clustering_pixel_accuracy = 0.0
clustering_IoU = 0.0
clustering_precision = 0.0
clustering_recall = 0.0
clustering_f_measure = 0.0
clustering_count = 0

region_growing_pixel_accuracy = 0.0
region_growing_IoU = 0.0
region_growing_precision = 0.0
region_growing_recall = 0.0
region_growing_f_measure = 0.0
region_growing_count = 0

region_splitting_pixel_accuracy = 0.0
region_splitting_IoU = 0.0
region_splitting_precision = 0.0
region_splitting_recall = 0.0
region_splitting_f_measure = 0.0
region_splitting_count = 0

def calculate_statistics(annotated_mask, segmented_mask):
    global global_pixel_accuracy
    global global_IoU
    global global_precision
    global global_recall
    global global_f_measure

    white_mask = (annotated_mask == 255)
    black_mask = (annotated_mask == 0)
    white_img = (segmented_mask == 255)
    black_img = (segmented_mask == 0)

    # correct detected pixels
    true_pozitive = np.sum(white_mask & white_img)
    # correct detected background pixels
    true_negative = np.sum(black_img & black_mask)
    # object detected pixels that are from background
    false_positive = np.sum(white_img & black_mask)
    # background detected pixels that are from object
    false_negative = np.sum(black_img & white_mask)

    total_pixels = true_pozitive + true_negative + false_positive + false_negative
    if total_pixels == 0:
        global_pixel_accuracy = 0
    else:
        global_pixel_accuracy = (true_pozitive + true_negative) / total_pixels

    denom_iou = true_pozitive + false_positive + false_negative
    if denom_iou == 0: # algoritm does not make any predictions (mostly the annotated mask has wrong format) 1 case in 200 images
        global_IoU = 0
    else:
        global_IoU = true_pozitive / denom_iou

    denom_precision = true_pozitive + false_positive
    if denom_precision == 0:
        global_precision = 0
    else:
        global_precision = true_pozitive / denom_precision

    denom_recall = true_pozitive + false_negative
    if denom_recall == 0:
        global_recall = 0
    else:
        global_recall = true_pozitive / denom_recall

    if global_precision + global_recall == 0:
        global_f_measure = 0
    else:
        global_f_measure = 2 * global_precision * global_recall / (global_precision + global_recall)


def show_statistics(annotated_mask, segmented_mask,caller=None):
    global global_pixel_accuracy,global_IoU,global_precision,global_recall,global_f_measure
    global gaussian_pixel_accuracy,gaussian_IoU,gaussian_precision,gaussian_recall,gaussian_f_measure,gaussian_count,watershed_pixel_accuracy
    global watershed_IoU,watershed_precision,watershed_recall,watershed_f_measure,watershed_count
    global clustering_pixel_accuracy,clustering_IoU,clustering_precision,clustering_recall,clustering_f_measure,clustering_count
    global region_growing_pixel_accuracy,region_growing_IoU,region_growing_precision,region_growing_recall,region_growing_f_measure,region_growing_count
    global region_splitting_pixel_accuracy,region_splitting_IoU,region_splitting_precision,region_splitting_recall,region_splitting_f_measure,region_splitting_count
    global max_accuracy,max_IoU,max_precision,max_recall,max_f_measure
    global max_accuracy_caller,max_IoU_caller,max_precision_caller,max_recall_caller, max_f_measure_caller
    global nr_gaussian_and_otsu_accuracies,nr_gaussian_and_otsu_IoUs,nr_gaussian_and_otsu_precisions,nr_gaussian_and_otsu_recalls,nr_gaussian_and_otsu_f_measures
    global nr_clustering_accuracies, nr_clustering_IoUs, nr_clustering_precisions,nr_clustering_recalls,nr_clustering_f_measures
    global nr_watershed_accuracies,nr_watershed_IoUs,nr_watershed_precisions,nr_watershed_recalls,nr_watershed_f_measures
    global nr_region_growing_accuracies,nr_region_growing_IoUs,nr_region_growing_precisions,nr_region_growing_recalls, nr_region_growing_f_measures
    global nr_region_splitting_accuracies,nr_region_splitting_IoUs,nr_region_splitting_precisions,nr_region_splitting_recalls,nr_region_splitting_f_measures

    calculate_statistics(annotated_mask,segmented_mask)
    if global_pixel_accuracy <0.3:
        segmented_mask =255-segmented_mask
        calculate_statistics(annotated_mask,segmented_mask)
    print(f"Pixel accuracy: {global_pixel_accuracy}")
    print(f"IoU: {global_IoU}")
    print(f"Precision: {global_precision}")
    print(f"Recall: {global_recall}")
    print(f"F-measure: {global_f_measure}")
    print("----------------------------------------")


    if caller ==1 and global_f_measure !=0 : #otsu and gaussian
        gaussian_pixel_accuracy += global_pixel_accuracy
        gaussian_IoU += global_IoU
        gaussian_precision += global_precision
        gaussian_recall += global_recall
        gaussian_f_measure += global_f_measure
        gaussian_count += 1
    if caller==2 and global_f_measure !=0: #watershed
        watershed_pixel_accuracy += global_pixel_accuracy
        watershed_IoU += global_IoU
        watershed_precision += global_precision
        watershed_recall += global_recall
        watershed_f_measure += global_f_measure
        watershed_count += 1
    if caller==3 and global_f_measure !=0: #clustering
        clustering_pixel_accuracy += global_pixel_accuracy
        clustering_IoU += global_IoU
        clustering_precision += global_precision
        clustering_recall += global_recall
        clustering_f_measure += global_f_measure
        clustering_count += 1
    if caller==4 and global_f_measure !=0: #region growing
        region_growing_pixel_accuracy += global_pixel_accuracy
        region_growing_IoU += global_IoU
        region_growing_precision += global_precision
        region_growing_recall += global_recall
        region_growing_f_measure += global_f_measure
        region_growing_count += 1
    if caller==5 and global_f_measure !=0: #region splitting
        region_splitting_pixel_accuracy += global_pixel_accuracy
        region_splitting_IoU += global_IoU
        region_splitting_precision += global_precision
        region_splitting_recall += global_recall
        region_splitting_f_measure += global_f_measure
        region_splitting_count += 1

    if max_accuracy < global_pixel_accuracy :
        max_accuracy = global_pixel_accuracy
        max_accuracy_caller=caller

    if max_IoU < global_IoU :
        max_IoU = global_IoU
        max_IoU_caller=caller

    if max_precision < global_precision :
        max_precision = global_precision
        max_precision_caller=caller

    if max_recall < global_recall :
        max_recall = global_recall
        max_recall_caller=caller

    if max_f_measure < global_f_measure :
        max_f_measure = global_f_measure
        max_f_measure_caller=caller

    if caller==5:
        if max_accuracy_caller==1:
            nr_gaussian_and_otsu_accuracies +=1
        if max_IoU_caller==1:
            nr_gaussian_and_otsu_IoUs +=1
        if max_precision_caller==1:
            nr_gaussian_and_otsu_precisions +=1
        if max_recall_caller==1:
            nr_gaussian_and_otsu_recalls +=1
        if max_f_measure_caller==1:
            nr_gaussian_and_otsu_f_measures +=1

        if max_accuracy_caller==2:
            nr_watershed_accuracies +=1
        if max_IoU_caller==2:
            nr_watershed_IoUs +=1
        if max_precision_caller==2:
            nr_watershed_precisions +=1
        if max_recall_caller==2:
            nr_watershed_recalls +=1
        if max_f_measure_caller==2:
            nr_watershed_f_measures +=1

        if max_accuracy_caller==3:
            nr_clustering_accuracies +=1
        if max_IoU_caller==3:
            nr_clustering_IoUs +=1
        if max_precision_caller==3:
            nr_clustering_precisions +=1
        if max_recall_caller==3:
            nr_clustering_recalls +=1
        if max_f_measure_caller==3:
            nr_clustering_f_measures +=1

        if max_accuracy_caller==4:
            nr_region_growing_accuracies +=1
        if max_IoU_caller==4:
            nr_region_growing_IoUs +=1
        if max_precision_caller==4:
            nr_region_growing_precisions +=1
        if max_recall_caller==4:
            nr_region_growing_recalls +=1
        if max_f_measure_caller==4:
            nr_region_growing_f_measures +=1

        if max_accuracy_caller==5:
            nr_region_splitting_accuracies +=1
        if max_IoU_caller==5:
            nr_region_splitting_IoUs +=1
        if max_precision_caller==5:
            nr_region_splitting_precisions +=1
        if max_recall_caller==5:
            nr_region_splitting_recalls +=1
        if max_f_measure_caller==5:
            nr_region_splitting_f_measures +=1

        max_accuracy = 0
        max_IoU = 0
        max_precision = 0
        max_recall = 0
        max_f_measure = 0
        max_accuracy_caller=0
        max_IoU_caller=0
        max_precision_caller=0
        max_recall_caller=0
        max_f_measure_caller=0


def load_and_display_image_and_annotated_mask(folder_images, folder_masks):
    global current_image
    global current_annotated_mask

    # List and sort files in both folders to ensure matching order
    image_files = sorted(os.listdir(folder_images))
    mask_files = sorted(os.listdir(folder_masks))

    for image_file, mask_file in zip(image_files, mask_files):
        # Construct full file paths
        image_path = os.path.join(folder_images, image_file)
        mask_path = os.path.join(folder_masks, mask_file)

        # Read image and mask
        current_image = cv2.imread(image_path)
        current_annotated_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        process_current_image_and_mask()

        # Display image and mask side by side
        cv2.imshow("Image", current_image)
        # cv2.imshow("Mask", current_annotated_mask)

        # gaussian -- green
        cv2.imshow("Processed Image Gaussian si OTSU", current_segmented_image_gaussian_and_otsu)
        # cv2.imshow("Processed Mask Gaussian si OTSU", current_segmented_mask_gaussian_and_otsu)

        # watershed -- magenta
        cv2.imshow("Processed Image Watershed",current_segmented_image_watershed)
        # cv2.imshow("Processed Mask Watershed",current_segmented_mask_watershed)

        # clustering with k-means -- cyan
        cv2.imshow("Processed Image Clustering",current_segmented_image_clustering)
        # cv2.imshow("Processed Mask Clustering",current_segmented_mask_clustering)

        # region_growing -- indigo
        cv2.imshow("Processed Image Region Growing",current_segmented_image_region_growing)
        # cv2.imshow("Processed Mask Region Growing",current_segmented_mask_region_growing)

        # region_splitting -- pink
        cv2.imshow("Processed Image Region Splitting",current_segmented_image_region_splitting)
        # cv2.imshow("Processed Mask Region Splitting",current_segmented_mask_region_splitting)

        print(f"Displaying: {image_file} and {mask_file}")
        print("*******************************************")
        key = cv2.waitKey(0)  # Wait for a key press to move to the next pair
        if key == 27:  # Press 'Esc' to exit early
            break

    cv2.destroyAllWindows()


def gaussian_and_otsu_thresholding():
    print("Gaussian and Otsu thresholding")
    global current_segmented_image_gaussian_and_otsu
    global current_segmented_mask_gaussian_and_otsu
    global current_image, current_annotated_mask

    grey_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized_image=clahe.apply(grey_image)
    blurred_image=cv2.GaussianBlur(equalized_image,(5,5),0)
    _, current_segmented_mask_gaussian_and_otsu = cv2.threshold(blurred_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply contours of the mask onto the segmented image
    contours, _ = cv2.findContours(current_segmented_mask_gaussian_and_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    current_segmented_image_gaussian_and_otsu = current_image.copy()
    cv2.drawContours(current_segmented_image_gaussian_and_otsu, contours, -1, (0, 255, 0), 2)

    show_statistics(current_annotated_mask, current_segmented_mask_gaussian_and_otsu,1)


def watershed_segmentation():
    print("Watershed segmentation")
    global current_image, current_annotated_mask
    global current_segmented_image_watershed,current_segmented_mask_watershed

    gray_image=cv2.cvtColor(current_image,cv2.COLOR_BGR2GRAY)
    _, segmented_with_otsu=cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.array_equal(current_image[:, :, 0], current_image[:, :, 1]) and np.array_equal(current_image[:, :, 1], current_image[:, :, 2]):
         segmented_with_otsu = 255 - segmented_with_otsu
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    closed_image=cv2.morphologyEx(segmented_with_otsu,cv2.MORPH_CLOSE,kernel)
    # cv2.imshow("Closed Image", closed_image)
    opened_image=cv2.morphologyEx(closed_image,cv2.MORPH_OPEN,kernel)
    kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilated_image=cv2.morphologyEx(opened_image,cv2.MORPH_DILATE,kernel1)
    # cv2.imshow("Dilated Image", dilated_image)
    distance_transform_image=cv2.distanceTransform(opened_image,cv2.DIST_L2,3)
    #normalized_distance_transform_image=cv2.normalize(distance_transform_image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    # cv2.imshow("Distance Transform", normalized_distance_transform_image)
    min_value,max_value,_,_=cv2.minMaxLoc(distance_transform_image)
    threshold_value=max_value*0.8
    _, sure_interior = cv2.threshold(distance_transform_image,threshold_value,255,cv2.THRESH_BINARY)
    sure_interior=sure_interior.astype(np.uint8)
    # cv2.imshow("Sure Interior", sure_interior)
    num_labels,labels=cv2.connectedComponents(sure_interior,connectivity=8,ltype=cv2.CV_32S)
    #grayscale_labels=(labels*(255.0/(num_labels-1))).astype(np.uint8)
    # cv2.imshow("Connected Components Labels", grayscale_labels)
    uncertainty_mask=dilated_image.copy()
    uncertainty_mask[sure_interior==255]=0
    # cv2.imshow("Uncertainty Mask", uncertainty_mask)
    markers=labels+1
    markers[uncertainty_mask!=0]=0
    markers_display=cv2.normalize(markers,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    #markers_color=cv2.cvtColor(markers_display,cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Markers (for Watershed)", markers_color)
    current_colored_image=cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR)
    cv2.watershed(current_colored_image,markers)
    current_segmented_image_watershed=current_image.copy()
    current_segmented_image_watershed[markers==-1]=[255,0,255]
    markers_display= cv2.normalize(markers,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    # cv2.imshow("Markers", markers_display)
    current_segmented_mask_watershed=markers_display.copy()
    current_segmented_mask_watershed=cv2.equalizeHist(current_segmented_mask_watershed)

    show_statistics(current_annotated_mask, current_segmented_mask_watershed,2)


def clustering_segmentation():
    global current_image, current_annotated_mask
    global current_segmented_image_clustering,current_segmented_mask_clustering
    print("Clustering segmentation using K-means")
    #preparing pixels for clustering
    pixel_values=current_image.reshape((-1,3))
    pixel_values=np.float32(pixel_values)
    #k_means
    kmeans=KMeans(n_clusters=3,random_state=0).fit(pixel_values)
    labels=kmeans.labels_
    #creating segmented image
    segmented_image=labels.reshape((current_image.shape[:2]))
    # obtaining the mask
    cluster_label=1
    current_segmented_mask_clustering=(segmented_image==cluster_label).astype(np.uint8)*255
    current_segmented_mask_clustering=cv2.medianBlur(current_segmented_mask_clustering,3)
    contours, _ = cv2.findContours(current_segmented_mask_clustering, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    current_segmented_image_clustering=current_image.copy()
    cv2.drawContours(current_segmented_image_clustering, contours, -1, (255, 255, 0), 2)
    show_statistics(current_annotated_mask, current_segmented_mask_clustering,3)

def region_growing_segmentation():
    print("Region growing segmentation")
    global current_image, current_annotated_mask
    global current_segmented_image_region_growing,current_segmented_mask_region_growing

    gray_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)
    height, width = gray_image.shape
    current_segmented_mask_region_growing = np.zeros_like(gray_image, dtype=np.uint8)
    visited = np.zeros_like(gray_image, dtype=np.bool_)
    _, binary_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(binary_img)

    for label in range(1, num_labels):
        seed_points = np.argwhere(labels == label)
        stack = [(pt[1], pt[0]) for pt in seed_points]
        while stack:
            x, y = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            intensity = gray_image[y, x]
            if abs(int(np.int16(intensity) - np.int16(gray_image[seed_points[0][0], seed_points[0][1]]))) <=10 :
                current_segmented_mask_region_growing[y, x] = 255
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        stack.append((nx, ny))

    contours, _ = cv2.findContours(current_segmented_mask_region_growing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    current_segmented_image_region_growing = current_image.copy()
    cv2.drawContours(current_segmented_image_region_growing, contours, -1, (255, 0, 0), 2)
    show_statistics(current_annotated_mask, current_segmented_mask_region_growing,4)


def region_splitting_segmentation():
    print("Region splitting segmentation")
    global current_image, current_annotated_mask
    global current_segmented_image_region_splitting, current_segmented_mask_region_splitting

    # Conversie la grayscale pentru procesare
    grayscale_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayscale_image=clahe.apply(grayscale_image)
    h, w = grayscale_image.shape[:2]
    segmented_img = np.zeros_like(grayscale_image, dtype=np.uint8)
    regions = [(0, 0, w, h)]

    def split_region(x, y, width, height):
        region = grayscale_image[y:y + height, x:x + width]
        std_dev = np.std(region)
        if std_dev > 15:
            return [
                (x, y, width // 2, height // 2),
                (x + width // 2, y, width // 2, height // 2),
                (x, y + height // 2, width // 2, height // 2),
                (x + width // 2, y + height // 2, width // 2, height // 2)
            ]
        return [(x, y, width, height)]

    def merge_region(x, y, width, height):
        region = grayscale_image[y:y + height, x:x + width]
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

    current_segmented_mask_region_splitting = cv2.morphologyEx(segmented_img, cv2.MORPH_CLOSE, kernel)
    current_segmented_mask_region_splitting = cv2.threshold(current_segmented_mask_region_splitting, 205, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(current_segmented_mask_region_splitting, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Creează o copie a imaginii color originale pentru afișarea contururilor
    current_segmented_image_region_splitting = current_image.copy()
    cv2.drawContours(current_segmented_image_region_splitting, contours, -1, (255, 155, 255), 2)
    show_statistics(current_annotated_mask, current_segmented_mask_region_splitting,5)


def process_current_image_and_mask():
    gaussian_and_otsu_thresholding()
    watershed_segmentation()
    clustering_segmentation()
    region_growing_segmentation()
    region_splitting_segmentation()


def main():
    global gaussian_pixel_accuracy,gaussian_IoU,gaussian_precision,gaussian_recall,gaussian_f_measure,gaussian_count
    global watershed_pixel_accuracy,watershed_IoU,watershed_precision,watershed_recall,watershed_f_measure,watershed_count
    global clustering_pixel_accuracy,clustering_IoU,clustering_precision,clustering_recall,clustering_f_measure,clustering_count
    global region_growing_pixel_accuracy,region_growing_IoU,region_growing_precision,region_growing_recall,region_growing_f_measure,region_growing_count
    global region_splitting_pixel_accuracy,region_splitting_IoU,region_splitting_precision,region_splitting_recall,region_splitting_f_measure,region_splitting_count
    global nr_gaussian_and_otsu_accuracies,nr_gaussian_and_otsu_IoUs,nr_gaussian_and_otsu_precisions,nr_gaussian_and_otsu_recalls, nr_gaussian_and_otsu_f_measures
    global nr_clustering_accuracies,nr_clustering_IoUs,nr_clustering_precisions,nr_clustering_recalls,nr_clustering_f_measures
    global nr_watershed_accuracies,nr_watershed_IoUs,nr_watershed_precisions, nr_watershed_recalls,nr_watershed_f_measures
    global nr_region_growing_accuracies,nr_region_growing_IoUs,nr_region_growing_precisions,nr_region_growing_recalls,nr_region_growing_f_measures
    global nr_region_splitting_accuracies,nr_region_splitting_IoUs,nr_region_splitting_precisions, nr_region_splitting_recall,nr_region_splitting_f_measures

    # Paths to the final folders
    final_images_folder = "finalImages"
    final_masks_folder = "final_masks"
    # Call the function to read and display images and masks
    load_and_display_image_and_annotated_mask(final_images_folder, final_masks_folder)


    print("Done!")
    gaussian_pixel_accuracy = gaussian_pixel_accuracy / gaussian_count
    gaussian_IoU = gaussian_IoU / gaussian_count
    gaussian_precision = gaussian_precision / gaussian_count
    gaussian_recall = gaussian_recall / gaussian_count
    gaussian_f_measure = gaussian_f_measure / gaussian_count
    print("gaussian pixel accuracy: ",gaussian_pixel_accuracy)
    print("gaussian Iou ",gaussian_IoU)
    print("gaussian precision: ",gaussian_precision)
    print("gaussian recall: ",gaussian_recall)
    print("gaussian f-measure: ",gaussian_f_measure)
    print("gaussian count: ",gaussian_count)

    watershed_pixel_accuracy = watershed_pixel_accuracy / watershed_count
    watershed_IoU = watershed_IoU / watershed_count
    watershed_precision = watershed_precision / watershed_count
    watershed_recall = watershed_recall / watershed_count
    watershed_f_measure = watershed_f_measure / watershed_count
    print("watershed pixel accuracy: ",watershed_pixel_accuracy)
    print("watershed IoU ",watershed_IoU)
    print("watershed precision: ",watershed_precision)
    print("watershed recall: ",watershed_recall)
    print("watershed f-measure: ",watershed_f_measure)
    print("watershed count: ",watershed_count)

    clustering_pixel_accuracy = clustering_pixel_accuracy / clustering_count
    clustering_IoU = clustering_IoU / clustering_count
    clustering_precision = clustering_precision / clustering_count
    clustering_recall = clustering_recall / clustering_count
    clustering_f_measure = clustering_f_measure / clustering_count
    print("clustering pixel accuracy: ",clustering_pixel_accuracy)
    print("clustering IoU ",clustering_IoU)
    print("clustering precision: ",clustering_precision)
    print("clustering recall: ",clustering_recall)
    print("clustering f-measure: ",clustering_f_measure)
    print("clustering count: ",clustering_count)

    region_growing_pixel_accuracy = region_growing_pixel_accuracy / region_growing_count
    region_growing_IoU = region_growing_IoU / region_growing_count
    region_growing_precision = region_growing_precision / region_growing_count
    region_growing_recall = region_growing_recall / region_growing_count
    region_growing_f_measure = region_growing_f_measure / region_growing_count
    print("region growing pixel accuracy: ",region_growing_pixel_accuracy)
    print("region growing IoU ",region_growing_IoU)
    print("region growing precision: ",region_growing_precision)
    print("region growing recall: ",region_growing_recall)
    print("region growing f-measure: ",region_growing_f_measure)
    print("region growing count: ",region_growing_count)

    region_splitting_pixel_accuracy = region_splitting_pixel_accuracy / region_splitting_count
    region_splitting_IoU = region_splitting_IoU / region_splitting_count
    region_splitting_precision = region_splitting_precision / region_splitting_count
    region_splitting_recall = region_splitting_recall / region_splitting_count
    region_splitting_f_measure = region_splitting_f_measure / region_splitting_count
    print("region splitting pixel accuracy: ",region_splitting_pixel_accuracy)
    print("region splitting IoU ",region_splitting_IoU)
    print("region splitting precision: ",region_splitting_precision)
    print("region splitting recall: ",region_splitting_recall)
    print("region splitting f-measure: ",region_splitting_f_measure)
    print("region splitting count: ",region_splitting_count)
    print("*******************************************")
    print("---------------------------")
    print("nr of gaussian and otsu accuracies: ",nr_gaussian_and_otsu_accuracies)
    print("nr of gaussian and otsu IoUs: ",nr_gaussian_and_otsu_IoUs)
    print("nr of gaussian and otsu precisions: ",nr_gaussian_and_otsu_precisions)
    print("nr of gaussian and otsu recalls: ",nr_gaussian_and_otsu_recalls)
    print("nr of gaussian and otsu f-measures: ",nr_gaussian_and_otsu_f_measures)
    print("nr of clustering accuracies: ", nr_clustering_accuracies)
    print("nr of clustering IoUs: ",nr_clustering_IoUs)
    print("nr of clustering precisions: ",nr_clustering_precisions)
    print("nr of clustering recalls: ",nr_clustering_recalls)
    print("nr of clustering f-measures: ",nr_clustering_f_measures)
    print("nr of watershed accuracies: ",nr_watershed_accuracies)
    print("nr of watershed IoUs: ",nr_watershed_IoUs)
    print("nr of watershed precisions: ",nr_watershed_precisions)
    print("nr of watershed recalls: ",nr_watershed_recalls)
    print("nr of watershed f-measures: ",nr_watershed_f_measures)
    print("nr of region growing accuracies: ",nr_region_growing_accuracies)
    print("nr of region growing IoUs: ",nr_region_growing_IoUs)
    print("nr of region growing precisions: ",nr_region_growing_precisions)
    print("nr of region growing recalls: ",nr_region_growing_recalls)
    print("nr of region growing f-measures: ",nr_region_growing_f_measures)
    print("nr of region splitting accuracies: ",nr_region_splitting_accuracies)
    print("nr of region splitting IoUs: ",nr_region_splitting_IoUs)
    print("nr of region splitting precisions: ",nr_region_splitting_precisions)
    print("nr of region splitting recalls: ",nr_region_splitting_recalls)
    print("nr of region splitting f-measures: ",nr_region_splitting_f_measures)
    print("*******************************************")

if __name__ == "__main__":
    main()