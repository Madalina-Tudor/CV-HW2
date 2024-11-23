import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

def load_image(image_path):
    """Load an image and convert it to a NumPy array."""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def save_image(image_array, output_path):
    """Save a NumPy array as an image."""
    image = Image.fromarray(np.uint8(image_array))
    image.save(output_path)

def rgb_to_hsv(image):
    """Convert an RGB image to HSV."""
    image = image.astype('float32') / 255.0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    cmax = np.max(image, axis=2)
    cmin = np.min(image, axis=2)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    # Hue calculation
    mask = delta != 0

    r_mask = (cmax == r) & mask
    g_mask = (cmax == g) & mask
    b_mask = (cmax == b) & mask

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4

    h = h / 6.0  # Normalize hue to [0, 1]

    s[mask] = delta[mask] / cmax[mask]
    s[cmax == 0] = 0  # If cmax is zero, saturation is zero

    hsv_image = np.stack((h, s, v), axis=-1)
    return hsv_image

def create_color_mask(hsv_image, lower, upper):
    """Create a mask for a specific color range."""
    mask = np.all(hsv_image >= lower, axis=2) & np.all(hsv_image <= upper, axis=2)
    return mask.astype('float')

def filter_large_objects(mask, min_size=500):
    """Filter out small connected components, keeping only large objects."""
    labels, num_labels = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(num_labels + 1))
    large_mask = np.zeros_like(mask)

    for i in range(1, num_labels + 1):
        if sizes[i] >= min_size:
            large_mask[labels == i] = 1

    return large_mask

def draw_rectangle_on_image(image, mask):
    """Draw rectangles around detected regions on the original image."""
    labels, num_labels = ndimage.label(mask)
    bounding_boxes = ndimage.find_objects(labels)

    pil_image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(pil_image)

    for box in bounding_boxes:
        if box is not None:
            y_slice, x_slice = box
            x1, x2 = x_slice.start, x_slice.stop
            y1, y2 = y_slice.start, y_slice.stop
            draw.rectangle([x1, y1, x2, y2], outline="red", width=10)

    return np.array(pil_image)

def detect_objects(image, output_dir, scale='normal'):
    hsv_image = rgb_to_hsv(image)

    if scale == 'small':
        min_size_orange = 1000  # Smaller min size for orange roof at small scale
        min_size_helicopter = 100  # Reduced min size for helicopter landing at small scale
    else:
        min_size_orange = 5000
        min_size_helicopter = 2000

    # 1) Blue pool detection
    if scale == 'medium' or scale == 'large':
        # Narrow the detection range for medium and large scales to exclude solar panels
        blue_lower = np.array([0.50, 0.39, 0.4])  # Cyanish blue, narrower hue range
        blue_upper = np.array([0.65, 0.9, 0.8])  # Darker cyan-blue range
    else:
        blue_lower = np.array([0.50, 0.29, 0.3])  # range for small scale
        blue_upper = np.array([0.75, 1.0, 1.0])

    blue_mask = create_color_mask(hsv_image, blue_lower, blue_upper)
    blue_large_mask = filter_large_objects(blue_mask, min_size=500)
    image_with_pool = draw_rectangle_on_image(image, blue_large_mask)
    save_image(image_with_pool, os.path.join(output_dir, 'blue_pool_with_rectangle.jpg'))
    print("Blue pool with rectangle saved")

    # 2) Orange roof detection
    orange_lower = np.array([0.02, 0.4, 0.3])
    orange_upper = np.array([0.12, 1.0, 1.0])
    orange_mask = create_color_mask(hsv_image, orange_lower, orange_upper)
    orange_large_mask = filter_large_objects(orange_mask, min_size=min_size_orange)
    image_with_roof = draw_rectangle_on_image(image, orange_large_mask)
    save_image(image_with_roof, os.path.join(output_dir, 'orange_roof_with_rectangle.jpg'))
    print("Orange roof with rectangle saved")

    # 3) Helicopter landing site detection
    pink_lower = np.array([0.87, 0.25, 0.25])  # Adjusted lower bound for pale pinkish-red
    pink_upper = np.array([1.0, 1.0, 1.0])  # Upper bound for pale red
    helicopter_landing_mask = create_color_mask(hsv_image, pink_lower, pink_upper)
    helicopter_landing_large_mask = filter_large_objects(helicopter_landing_mask, min_size=min_size_helicopter)
    image_with_helicopter_landing = draw_rectangle_on_image(image, helicopter_landing_large_mask)
    save_image(image_with_helicopter_landing, os.path.join(output_dir, 'helicopter_landing_with_rectangle.jpg'))
    print("Helicopter landing site with rectangle saved")

def main():
    # Directory where images are stored
    images_dir = os.path.join('..', 'images')

    # Define the images for processing based on scale
    image_info = {
        'small': [os.path.join(images_dir, 'Gura_Portitei_Scara_010.jpg'),
                  os.path.join(images_dir, 'Gura_Portitei_Scara_020.jpg')],
        'medium': [os.path.join(images_dir, 'Gura_Portitei_Scara_040.jpg'),
                   os.path.join(images_dir, 'Gura_Portitei_Scara_080.jpg')],
        'large': [os.path.join(images_dir, 'Gura_Portitei_Scara_100.jpg')]
    }

    # Change the base output directory to "results/color_filter"
    base_output_dir = os.path.join('..', 'results', 'color_filter')

    for scale, image_paths in image_info.items():
        output_dir = os.path.join(base_output_dir, scale + "_scale")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in image_paths:
            image = load_image(image_file)
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            print(f"Processing {image_file} at scale: {scale}")
            detect_objects(image, output_dir, scale)

if __name__ == '__main__':
    main()
