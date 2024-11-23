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

def create_box_filter(size):
    """Create a box filter kernel of given size."""
    return np.ones((size, size)) / (size * size)

def create_gaussian_filter(size, sigma):
    """Create a Gaussian filter kernel of given size and sigma."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / np.sum(kernel)

def apply_filters(image, filter_sizes, output_dir, scale, image_name):
    """Apply box and Gaussian filters of different sizes to the image and save results."""
    for size in filter_sizes:
        print(f"\nApplying filters of size {size}x{size} to {image_name} at scale {scale}")

        # Box Filter
        box_kernel = create_box_filter(size)
        box_filtered = convolve(image, box_kernel)
        box_output_path = os.path.join(output_dir, f'{image_name}_box_filtered_{scale}_{size}.jpg')
        save_image(box_filtered, box_output_path)
        print(f"Box filter applied and saved as '{box_output_path}'")

        # Gaussian Filter
        sigma = size / 6.0
        gaussian_kernel = create_gaussian_filter(size, sigma)
        gaussian_filtered = convolve(image, gaussian_kernel)
        gaussian_output_path = os.path.join(output_dir, f'{image_name}_gaussian_filtered_{scale}_{size}.jpg')
        save_image(gaussian_filtered, gaussian_output_path)
        print(f"Gaussian filter applied and saved as '{gaussian_output_path}'")

def process_images(image_files, output_subdir, filter_sizes):
    """Process images with both box and Gaussian filters and save the results in subfolders."""
    # Define the base output directory as results/color_filter
    base_output_dir = os.path.join('..', 'results', 'gaussian_and_boxed_filter')

    # Create the base "gaussian_and_box_filter" directory if it doesn't exist
    output_dir = os.path.join(base_output_dir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]  # Get image name without extension
        image = load_image(image_file)
        print(f"Processing {image_file} with filters of sizes: {filter_sizes}")

        # Apply both filters and save the results
        apply_filters(image, filter_sizes, output_dir, output_subdir, image_name)

def main():
    # Directory where images are stored
    images_dir = os.path.join('..', 'images')

    # Define the images for processing based on scale
    image_info = {
        'small_scale': [
            os.path.join(images_dir, 'Gura_Portitei_Scara_010.jpg'),
            os.path.join(images_dir, 'Gura_Portitei_Scara_020.jpg'),
            os.path.join(images_dir, 'Gura_Portitei_Scara_0025.jpg')
        ],
        'medium_scale': [
            os.path.join(images_dir, 'Gura_Portitei_Scara_040.jpg'),
            os.path.join(images_dir, 'Gura_Portitei_Scara_080.jpg')
        ],
        'large_scale': [
            os.path.join(images_dir, 'Gura_Portitei_Scara_100.jpg')
        ]
    }

    # Filter sizes to test
    filter_sizes = [3, 5, 7]

    # Process each group of images and save them in the corresponding subfolder
    for output_subdir, image_files in image_info.items():
        process_images(image_files, output_subdir, filter_sizes)

if __name__ == '__main__':
    main()
