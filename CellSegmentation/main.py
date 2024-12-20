import cv2
import numpy as np
import case_functions as case_fun

def main():
    i = 1
    img = cv2.imread(rf'./CompositeMasks/Images/annotatedImages/{i}/images/1.png')
    # a se verifica masca anotata
    annotated_mask = cv2.imread(rf'./CompositeMasks/AnnotatedMasks/results_new_mask{i}.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("original", img)
    case_fun.case1(img.copy(), annotated_mask)
    case_fun.case2(img.copy(), annotated_mask)
    case_fun.case3(img.copy(), annotated_mask)
    case_fun.case4(img.copy(), annotated_mask)
    case_fun.case5(img.copy(), annotated_mask)
    case_fun.case6(img.copy(), annotated_mask)
    case_fun.case7(img.copy(), annotated_mask)
    # case_fun.case8(img.copy(), annotated_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
