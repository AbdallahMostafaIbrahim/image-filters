import numpy as np
from PIL import Image
import datetime

def apply_mean_blur(image_array, kernel_size=101):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return apply_blur(image_array, kernel)


def apply_blur(image_array, kernel):
    # Image and kernel dimensions
    height, width, channels = image_array.shape
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2  # Padding size

    # Add padding 
    padded_image = np.pad(image_array, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Output image
    blurred_image = np.zeros_like(image_array)

    # Loop over each pixel in the original image
    for y in range(height):  # Loop over image rows
        for x in range(width):  # Loop over image columns
            for c in range(channels):  # Loop over color channels
                neighbors = padded_image[y:y + kernel_size, x:x + kernel_size, c]

                pixel_value = np.sum(neighbors * kernel)

                blurred_image[y, x, c] = np.clip(pixel_value, 0, 255)

    return blurred_image.astype(np.uint8)


def generate_gaussian_kernel(kernel_size, sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd number and >= 3.")

    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError("Kernel sum is zero. Check sigma value.")
    kernel /= kernel_sum
    
    return kernel


def apply_gaussian_blur(image_array, kernel_size=31, sigma=5.17):
    # Generate the Gaussian kernel
    kernel = generate_gaussian_kernel(kernel_size, sigma)

    # Apply the kernel using the generic blur function
    return apply_blur(image_array, kernel)


def mirror_image(image_array, mode):
    # Image dimensions
    height, width, channels = image_array.shape

    # Transformation matrices
    if mode == 'horizontal':
        T = np.array([[-1,  0, width - 1],  # Flip x-axis and adjust for translation
                      [ 0,  1, 0],
                      [ 0,  0, 1]])
    elif mode == 'vertical':
        T = np.array([[ 1,  0, 0],
                      [ 0, -1, height - 1],  # Flip y-axis and adjust for translation
                      [ 0,  0, 1]])
    else:
        raise ValueError("Mode must be 'horizontal' or 'vertical'.")

    # Create an empty array for the transformed image
    transformed_image = np.zeros_like(image_array)

    for i in range(height):
        for j in range(width):
            original_coordinates = np.array([j, i, 1])
            
            # Transform the coordinate
            transformed_coordinates = np.dot(T, original_coordinates)
            x_new, y_new = int(transformed_coordinates[0]), int(transformed_coordinates[1])

            # Map the pixel value to the new position
            if 0 <= x_new < width and 0 <= y_new < height:
                transformed_image[y_new, x_new, :] = image_array[i, j, :]

    return transformed_image



image_path = input("Specify Image Path: ")

# Load the input image
input_image = Image.open(image_path)
image_array = np.array(input_image)

print("Choose the type of filter: \n1. Gaussian Blur\n2. Mean Blur\n3. Mirror Horizontally\n4. Mirror Vertically")
choice = int(input())

if choice == 1:
    kernel_size = int(input("Enter kernel size , usually 15 is a good choice: "))
    if kernel_size % 2 == 0:
        print("Kernel size must be an odd number.")
    blurred_image_array = apply_gaussian_blur(image_array, kernel_size=kernel_size, sigma=kernel_size/6)
elif choice == 2:
    kernel_size = int(input("Enter kernel size (provide an odd number, usually 15 is a good choice): "))
    if kernel_size % 2 == 0:
        print("Kernel size must be an odd number.")
    blurred_image_array = apply_mean_blur(image_array, kernel_size=kernel_size)
elif choice == 3:
    blurred_image_array = mirror_image(image_array, mode='horizontal')
elif choice == 4:
    blurred_image_array = mirror_image(image_array, mode='vertical')
else:
    print("Invalid choice!")
    exit(1)

# Convert the result back to an image
blurred_image = Image.fromarray(blurred_image_array)

# Save the output image with a timestamped filename day-month-year_hour-minute-second
current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
choiceToName = {1: "GaussianBlur", 2: "MeanBlur", 3: "MirrorHorizontal", 4: "MirrorVertical"}
# Get Image name and extension from path
image_name = image_path.split("/")[-1].split(".")[0]
image_extension = image_path.split("/")[-1].split(".")[1]
output_filename = f"{image_name}-{choiceToName[choice]}-{current_time}.{image_extension}"
blurred_image.save(output_filename)

print(f"Output image saved as {output_filename}")
