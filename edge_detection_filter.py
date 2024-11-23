import os
import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image and convert it to a NumPy array."""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def save_image(image_array, output_path):
    """Save a NumPy array as an image."""
    image = Image.fromarray(np.uint8(image_array))
    image.save(output_path)

def convolve(image, kernel):
    """Convolve an image with a kernel without using any libraries."""
    kernel_height, kernel_width = kernel.shape
    image_height, image_width, channels = image.shape

    # Pad the image
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'edge')

    # Initialize the output
    output = np.zeros((image_height, image_width, channels))

    # Perform convolution
    for y in range(image_height):
        for x in range(image_width):
            for c in range(channels):
                region = padded_image[y:y + kernel_height, x:x + kernel_width, c]
                output[y, x, c] = np.sum(region * kernel)
    return output

def detect_edges(image, output_dir, scale, image_name):
    """Detect edges using Sobel operator and custom Laplacian edge detection."""

    # Sobel operator kernels
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    # Convert image to grayscale
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Sobel Edge Detection (Horizontal and Vertical)
    grad_x = convolve(gray_image[..., np.newaxis], sobel_x)
    grad_y = convolve(gray_image[..., np.newaxis], sobel_y)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2).squeeze()
    grad_magnitude = (grad_magnitude / grad_magnitude.max()) * 255

    # Save Sobel Edge Detection Result in the respective subfolder
    sobel_output_path = os.path.join(output_dir, f'{image_name}_sobel_edges_{scale}.jpg')
    save_image(grad_magnitude, sobel_output_path)
    print(f"Sobel edge detection completed and saved as '{sobel_output_path}'")

    # Custom Laplacian Edge Detection
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

    edges = convolve(gray_image[..., np.newaxis], laplacian_kernel)
    edges = np.abs(edges).squeeze()
    edges = (edges / edges.max()) * 255

    # Save Custom Laplacian Edge Detection Result in the respective subfolder
    laplacian_output_path = os.path.join(output_dir, f'{image_name}_laplacian_edges_{scale}.jpg')
    save_image(edges, laplacian_output_path)
    print(f"Laplacian edge detection completed and saved as '{laplacian_output_path}'")

def process_images(image_files, output_subdir, base_output_dir="Edge Detection"):
    """Process images and apply edge detection filters, saving results in subfolders."""
    # Create the base "Edge Detection" directory if it doesn't exist
    output_dir = os.path.join(base_output_dir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        image = load_image(image_file)
        print(f"Processing {image_file} for edge detection")

        # Apply edge detection filters and save the results
        detect_edges(image, output_dir, output_subdir, image_name)

def main():
    # Define the images for processing based on scale
    image_info = {
        'small_scale': [
            'images/Gura_Portitei_Scara_010.jpg',
            'images/Gura_Portitei_Scara_020.jpg',
            'images/Gura_Portitei_Scara_0025.jpg'
        ],
        'medium_scale': [
            'images/Gura_Portitei_Scara_040.jpg',
            'images/Gura_Portitei_Scara_080.jpg'
        ],
        'large_scale': [
            'images/Gura_Portitei_Scara_100.jpg'
        ]
    }

    # Process each group of images and save them in the corresponding subfolder
    for output_subdir, image_files in image_info.items():
        process_images(image_files, output_subdir)

if __name__ == '__main__':
    main()
