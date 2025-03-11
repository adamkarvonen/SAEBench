from PIL import Image
import os


def create_two_column_layout(folder_paths, output_path):
    """
    Create a 2-column layout with 4 images on the left and 3 on the right,
    with white space filling the bottom-right space.
    """
    # List to store all images
    images = []

    # Load all images
    for folder in folder_paths:
        for file in os.listdir(folder):
            if file.endswith(".png"):
                img_path = os.path.join(folder, file)
                img = Image.open(img_path)
                images.append(img)

    # Get dimensions from first image
    width, height = images[0].size

    # Calculate dimensions for final image
    final_width = width * 2
    final_height = height * 4

    # Create new image with white background
    result = Image.new("RGB", (final_width, final_height), "white")

    # Paste first 4 images in left column
    for i in range(4):
        result.paste(images[i], (0, i * height))

    # Paste remaining 3 images in right column
    for i in range(4, 8):
        result.paste(images[i], (width, (i - 4) * height))

    # Save result
    result.save(output_path)
    print(f"Combined image saved to {output_path}")


# Example usage
base_folder = "./images_checkpoints"
folder_paths = [
    os.path.join(base_folder, "absorption"),
    os.path.join(base_folder, "autointerp"),
    os.path.join(base_folder, "core"),
    os.path.join(base_folder, "scr"),
    os.path.join(base_folder, "sparse_probing"),
    os.path.join(base_folder, "tpp"),
    os.path.join(base_folder, "unlearning"),
    os.path.join(base_folder, "ravel"),
]

output_path = os.path.join(base_folder, "all_checkpoints.png")
create_two_column_layout(folder_paths, output_path)
