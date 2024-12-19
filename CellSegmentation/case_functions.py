import cv2
import segmentation_functions as sf

def case1(img,annotated_mask):
    """ pentru imagini cu contrast global bun si zgomot redus"""

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray img", gray_img)
    contoured_img1, f_measure1 = sf.custom_segmentation(img, gray_img, annotated_mask)
    cv2.imshow('Custom segmented', contoured_img1)


def case2(img,annotated_mask):
    """ folosit cand contururile sunt clar delimitate, dar exista zgomot punctiform """
    # Run the improved watershed function
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    # cv2.imshow("")
    segmented_img, binary_img, markers = sf.improved_watershed(enhanced_img, annotated_mask)
    cv2.imshow('WaterShed segmented', segmented_img)

def case3(img,annotated_mask):
    """ separarea celulelor suprapuse """
    pass

def case4(img,annotated_mask):
    """ ideal pentru imagini cu contrast slab intre celule si fundal """
    pass

def case5(img,annotated_mask):
    """ eficient pentru detalii complexe cu detalii fine"""
    pass

def case6(img,annotated_mask):
    """ recomandat pentru imagini cu zgomot sau pentru imagini unde formele obiectelor sunt simple"""
    pass

def case7(img,annotated_mask):
    """ util pentru imagini slab iluminate si cu zgomot moderat """
    pass

def case8(img,annotated_mask):
    """ util pentru imagini cu variatii complexe in itensitate """
    pass