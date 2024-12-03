#! /usr/bin/env python3

import cv2
import matplotlib.pyplot as plt

def visualize_and_save_hmap(image_path, save_path):
    """
    Function to visualize the height map image and save it to disk.
    
    Args:
    image_path (str): Path to the height map image.
    save_path (str): Path to save the visualized image.
    
    Returns:
    None
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    
    # Display the image using matplotlib
    plt.imshow(image, cmap='viridis')
    plt.colorbar()
    plt.title('Height Map Visualization')
    
    # Save the visualized image to disk
    plt.savefig(save_path)
    plt.close()
    
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    
    image_path = "carla/carla/Town01.3/0__center__semantic_segmentation.png"
    save_path = "0__center__semantic_segmentation.png"
    
    visualize_and_save_hmap(image_path, save_path)

