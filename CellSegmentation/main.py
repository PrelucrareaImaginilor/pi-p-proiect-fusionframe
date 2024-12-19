import cv2
import numpy as np
import case_functions as cfun

def main():
    i = 4
    img = cv2.imread(rf'./CompositeMasks/Images/annotatedImages/{i}/images/1.png')
    annotated_mask = cv2.imread(rf'./CompositeMasks/AnnotatedMasks/results_new_mask{i}.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original", img)
    cfun.case1(img,annotated_mask)
    cfun.case2(img,annotated_mask)
    cfun.case3(img,annotated_mask)
    cfun.case4(img,annotated_mask)
    cfun.case5(img,annotated_mask)
    cfun.case6(img,annotated_mask)
    cfun.case7(img,annotated_mask)
    cfun.case8(img,annotated_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
