import cv2
import numpy as np
import segmentation_functions as sf

def case1(img,annotated_mask):
    """ pentru imagini cu contrast global bun si zgomot redus"""

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray img", gray_img)
    contoured_img1, f_measure1 = sf.custom_segmentation(img, gray_img, annotated_mask)
    cv2.imshow('Case1:Custom segmented', contoured_img1)


def case2(img,annotated_mask):
    """ folosit cand contururile sunt clar delimitate, dar exista zgomot punctiform """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold = 15  # Prag de similaritate
    region_growing_result, f_measure_rg = sf.region_growing(gray_img, annotated_mask, threshold)
    print(f"Region Growing f_measure: {f_measure_rg}")
    #cv2.imshow("Region Growing (Segmented)", region_growing_result)
    contours, _ = cv2.findContours(region_growing_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Draw contours on the original image
    output_img = img.copy()
    cv2.drawContours(output_img, contours, -1, (0, 0, 255), 2)  # Red contours with thickness 2

    # Step 5: Display the result
    cv2.imshow("Case2:Region Growing (Segmented with Contours)", output_img)

def case3(img,annotated_mask):
    """ separarea celulelor suprapuse """

    # Convert to grayscale for processing
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply Bilateral Filter
    # filtered_img = cv2.bilateralFilter(gray_img, d=9, sigmaColor=75, sigmaSpace=75)
    #
    #
    # # Apply Sobel Edge Detection
    # sobelx = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(filtered_img, cv2.CV_64F, 0, 1, ksize=5)
    # sobel_combined = cv2.magnitude(sobelx, sobely).astype(np.uint8)
    #
    # sobel_normalized = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    # combined_image = cv2.addWeighted(filtered_img, 0.7, sobel_normalized, 0.3, 0)
    # cv2.imshow('Grayscale Preprocessed Mask', combined_image )

    segmented_img, binary_img, markers = sf.improved_watershed(gray_img, annotated_mask)

    # Draw contours on the original color image
    final_result = img.copy()
    final_result[markers == -1] = [0, 0, 255]  # Draw red contours on the original image

    # # Display results
    cv2.imshow('Case3:Watershed+Bilateral+Sobel', segmented_img)
    #cv2.imshow('Grayscale Preprocessed Mask', binary_img)

def case4(img,annotated_mask):
    """ ideal pentru imagini cu contrast slab intre celule si fundal """
    segmented_img, refined_mask, f_measure = sf.graph_cut_segmentation(img, annotated_mask)
    cv2.imshow("Case4:Graph Cut Segmented Image", segmented_img)
   # cv2.imshow("Graph Cut Binary Mask", refined_mask)


def case5(img,annotated_mask):
    """ eficient pentru detalii complexe cu detalii fine"""
    # Step 1: Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian smoothing to reduce noise
    smoothed_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Step 3: Perform Laplacian edge detection
    laplacian_edges = cv2.Laplacian(smoothed_img, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian_edges)  # Convert to 8-bit image

    # Step 4: Apply multiple thresholding
    thresholds = [5, 10, 15, 20]  # Example threshold values
    binary_images = [cv2.threshold(laplacian_edges, t, 255, cv2.THRESH_BINARY)[1] for t in thresholds]

    # Combine all thresholds using a logical OR operation
    combined_binary = np.zeros_like(binary_images[0])
    for binary in binary_images:
        combined_binary = cv2.bitwise_or(combined_binary, binary)

    # Step 5: Optional - Enhance contrast before thresholding (using CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(smoothed_img)

    # Step 6: Find contours from the combined binary image
    contours, _ = cv2.findContours(combined_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 8: Draw contours on the original image
    output_img = img.copy()
    cv2.drawContours(output_img, contours, -1, (0, 0, 255), 1)  # Red contours with thickness 2

    # (Optional) Display results for debugging
    # cv2.imshow('Grayscale Image', gray_img)
    # cv2.imshow('Smoothed Image', smoothed_img)
    # cv2.imshow('Laplacian Edges', laplacian_edges)
    # cv2.imshow('Combined Thresholding', combined_binary)
    # cv2.imshow('Enhanced Image (CLAHE)', enhanced_img)
    cv2.imshow('Case5:Contours on Original Image', output_img)


def case6(img,annotated_mask):
    """ recomandat pentru imagini cu zgomot sau pentru imagini unde formele obiectelor sunt simple"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    final_segmented_img = sf.region_splitting_and_merging(gray_img, threshold=15, min_size=20)
    #f_measure = hf.statistics(annotated_mask, final_segmented_img)
    cv2.imshow("Case6:Segmented Image (Region Splitting and Merging)", final_segmented_img)

def case7(img,annotated_mask):
    """ util pentru imagini slab iluminate si cu zgomot moderat """
    # Step 1: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # Step 2: Apply Laplacian of Gaussian (LoG) for edge detection
    blurred_img = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Apply Laplacian operator to find edges
    laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_64F)

    # Convert Laplacian image back to uint8 for thresholding
    laplacian_img = np.uint8(np.abs(laplacian_img))

    # Step 3: Apply Thresholding to segment the image
    _, binary_img = cv2.threshold(laplacian_img, 30, 255, cv2.THRESH_BINARY)

    # Optionally, apply morphological operations to clean up the result
    kernel = np.ones((3, 3), np.uint8)
    cleaned_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Step 4: Find contours on the cleaned binary image
    contours, _ = cv2.findContours(cleaned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Draw the contours on the original image
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Green color and thickness of 2

    # Display the original image with contours
    cv2.imshow("Case7:Image with Segmentation Contours", img_with_contours)

def case8(img,annotated_mask):
    """ util pentru imagini cu variatii complexe in itensitate """
    pass