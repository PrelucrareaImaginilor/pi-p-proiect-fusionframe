import os
import cv2
import numpy as np


def assemble_masks(mask_folder):
    # Initialize a blank mask to combine all masks
    combined_mask = None

    for mask_file in sorted(os.listdir(mask_folder)):
        mask_path = os.path.join(mask_folder, mask_file)
        if not os.path.isfile(mask_path):
            continue

        # Read the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if combined_mask is None:
            combined_mask = np.zeros_like(mask)

        # Combine masks by taking the maximum pixel value
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    return combined_mask

def process_folders(base_folder):
    image_counter = 1

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        images_folder = os.path.join(subfolder_path, "images")
        masks_folder = os.path.join(subfolder_path, "masks")

        # Process image folder
        for image_file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_file)
            if not os.path.isfile(image_path):
                continue

            # Copy the image to finalImages with a new name
            new_image_name = f"image_{image_counter}.png"
            new_image_path = os.path.join("finalImages", new_image_name)
            image = cv2.imread(image_path)
            cv2.imwrite(new_image_path, image)

            # Process masks folder and create a final mask
            combined_mask = assemble_masks(masks_folder)
            if combined_mask is not None:
                final_mask_name = f"final_mask_{image_counter}.png"
                final_mask_path = os.path.join("final_masks", final_mask_name)
                cv2.imwrite(final_mask_path, combined_mask)

            image_counter += 1
            print(f"Processed image {image_counter}")

if __name__ == "__main__":
    base_folder = "./stage1_train"
    process_folders(base_folder)
