import cv2
import numpy as np
import os
def main():
    directory_path = r'./Images/annotatedImages'
    file_list = os.listdir(directory_path)
    file_count = sum(1 for file in file_list if os.path.isdir(os.path.join(directory_path, file)))
    for j in range(1, file_count+1):
        directory_path1 = rf'./Images/annotatedImages/{j}/masks'
        file_list1 = os.listdir(directory_path1)
        file_count_f = sum(1 for file in file_list1 if os.path.isfile(os.path.join(directory_path1, file)))
        auxing=cv2.imread(rf'./Images/annotatedImages/{j}/masks/1.png')
        for i in range(2,file_count_f):
            img=cv2.imread(rf'./Images/annotatedImages/{j}/masks/{i}.png')
            auxing=cv2.add(auxing, img)
        cv2.imwrite(rf'results_new_mask{j}.png',auxing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
     #salvare fisier
     #png inapoi pe disc
if __name__ == "__main__":
    main()
