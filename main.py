import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, size, color=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color

        self.top_left = None
        self.top_right = None
        self.bottom_left = None
        self.bottom_right = None

    def is_leaf(self):
        return self.top_left is None and self.top_right is None and self.bottom_left is None and self.bottom_right is None




def read_csv_to_image(csv_file):
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        pixels = next(reader)  # First line: pixel indices
        colors = next(reader)  # Second line: RGB values as strings

    # Convert colors to a NumPy array
    rgb_values = np.array([list(map(int, color.strip('"').split(','))) for color in colors])

    # Determine the size of the image (e.g., 4x4)
    size = int(len(pixels) ** 0.5)

    # Reshape the 1D array into a 2D image
    image_array = rgb_values.reshape((size, size, 3))
    return image_array





def load_image(file_path):
    image = Image.open(file_path).convert('RGB')
    return np.array(image)




def save_image(image_array, file_name):
    image = Image.fromarray(image_array)
    image.save(file_name)





def is_uniform_color(image, x, y, size):
    color = image[y, x]
    for i in range(y, y + size):
        for j in range(x, x + size):
            if not np.array_equal(image[i, j], color):
                return False
    return True







def build_quadtree(image, x, y, size):
    if size == 1:
        return Node(x, y, size, color=tuple(image[y, x]))

    half_size = size // 2
    top_left = (x, y)
    top_right = (x + half_size, y)
    bottom_left = (x, y + half_size)
    bottom_right = (x + half_size, y + half_size)

    node = Node(x, y, size)

    if is_uniform_color(image, *top_left, half_size):
        node.top_left = Node(*top_left, half_size, color=tuple(image[top_left[1], top_left[0]]))
    else:
        node.top_left = build_quadtree(image, *top_left, half_size)

    if is_uniform_color(image, *top_right, half_size):
        node.top_right = Node(*top_right, half_size, color=tuple(image[top_right[1], top_right[0]]))
    else:
        node.top_right = build_quadtree(image, *top_right, half_size)

    if is_uniform_color(image, *bottom_left, half_size):
        node.bottom_left = Node(*bottom_left, half_size, color=tuple(image[bottom_left[1], bottom_left[0]]))
    else:
        node.bottom_left = build_quadtree(image, *bottom_left, half_size)

    if is_uniform_color(image, *bottom_right, half_size):
        node.bottom_right = Node(*bottom_right, half_size, color=tuple(image[bottom_right[1], bottom_right[0]]))
    else:
        node.bottom_right = build_quadtree(image, *bottom_right, half_size)

    return node





def display_image(node, target_size):
    if node.size < target_size:
        raise ValueError(f"Target size {target_size} is larger than the root node size {node.size}.")

    image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    def fill_image_preorder(node):
        if node.is_leaf():
            start_x = node.x
            start_y = node.y
            size = node.size

            for i in range(start_y, start_y + size):
                for j in range(start_x, start_x + size):
                    image[i, j] = node.color
        else:
            if node.top_left:
                fill_image_preorder(node.top_left)
            if node.top_right:
                fill_image_preorder(node.top_right)
            if node.bottom_left:
                fill_image_preorder(node.bottom_left)
            if node.bottom_right:
                fill_image_preorder(node.bottom_right)

    fill_image_preorder(node)

    plt.imshow(image)
    plt.axis('off')
    plt.show()





def searchSubspacesWithRange(node, x1, y1, x2, y2):
    
    if node.is_leaf():
        start_x = node.x
        start_y = node.y
        end_x = start_x + node.size
        end_y = start_y + node.size
        
        if (start_x < x2 and end_x > x1) and (start_y < y2 and end_y > y1):
            return node
        else:
            node.color = (255, 255, 255)
            return node
    
    
    if node.top_left:
        node.top_left = searchSubspacesWithRange(node.top_left, x1, y1, x2, y2)
        
    if node.top_right:
        node.top_right = searchSubspacesWithRange(node.top_right, x1, y1, x2, y2)
        
    if node.bottom_left:
        node.bottom_left = searchSubspacesWithRange(node.bottom_left, x1, y1, x2, y2)
        
    if node.bottom_right:
        node.bottom_right = searchSubspacesWithRange(node.bottom_right, x1, y1, x2, y2)

    
    if node.top_left == None and node.top_right == None and node.bottom_left == None and node.bottom_right == None:
        return None


    return node








def compress_image(node, target_size):
    if node.size < target_size:
        raise ValueError(f"Target size {target_size} is larger than the root node size {node.size}.")

    if node.size % target_size != 0:
        raise ValueError(f"Root node size {node.size} must be divisible by target size {target_size}.")

    compressed_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    scale_factor = node.size // target_size

    def fill_image_preorder(node):
        if node.is_leaf():
            start_x = node.x // scale_factor
            start_y = node.y // scale_factor
            size = node.size // scale_factor

            for i in range(start_y, start_y + size):
                for j in range(start_x, start_x + size):
                    compressed_image[i, j] = node.color
        else:
            if node.top_left:
                fill_image_preorder(node.top_left)
            if node.top_right:
                fill_image_preorder(node.top_right)
            if node.bottom_left:
                fill_image_preorder(node.bottom_left)
            if node.bottom_right:
                fill_image_preorder(node.bottom_right)

    fill_image_preorder(node)

    return compressed_image






def TreeDepth(node):
    if node.is_leaf():
        return 0

    depths = []
    if node.top_left:
        depths.append(TreeDepth(node.top_left))
    if node.top_right:
        depths.append(TreeDepth(node.top_right))
    if node.bottom_left:
        depths.append(TreeDepth(node.bottom_left))
    if node.bottom_right:
        depths.append(TreeDepth(node.bottom_right))

    return 1 + max(depths) if depths else 0





def pixelDepth(node, x, y, current_depth=0):
    # If the node is a leaf, check if the pixel matches the node's first pixel
    if node.is_leaf():
        start_x = node.x
        start_y = node.y
        if start_x <= x < start_x + node.size and start_y <= y < start_y + node.size:
            return current_depth
        return -1  # Pixel is not in this node

    # Determine the size of each child quadrant
    half_size = node.size // 2

    # Check which quadrant the pixel belongs to and recurse into that quadrant
    if x < node.x + half_size and y < node.y + half_size:
        return pixelDepth(node.top_left, x, y, current_depth + 1)
    elif x >= node.x + half_size and y < node.y + half_size:
        return pixelDepth(node.top_right, x, y, current_depth + 1)
    elif x < node.x + half_size and y >= node.y + half_size:
        return pixelDepth(node.bottom_left, x, y, current_depth + 1)
    elif x >= node.x + half_size and y >= node.y + half_size:
        return pixelDepth(node.bottom_right, x, y, current_depth + 1)

    # If none of the conditions are met, return -1 (pixel not found)
    return -1




# Loading image
image = load_image("Hidson-1.png")


# Example: Replace this with the path to your CSV file
# csv_file_path = "image2_RGB.csv"

# Read the CSV file and convert it to a 2D NumPy array
# image = read_csv_to_image(csv_file_path)


height, width, _ = image.shape

if height == width:
    quadtree = build_quadtree(image, 0, 0, height)




# Calculate and print tree depth
depth = TreeDepth(quadtree)
print(f"The depth of the quadtree is: {depth}")


# Set the target size
target_size_for_compress = 4

# Check and compress the image
try:
    compressed_image = compress_image(quadtree, target_size_for_compress)
    save_image(compressed_image, "compressed_image_from_tree_preorder.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.title("Compressed Image")
    plt.show()
except ValueError as e:
    print(f"Error: {e}")



# x, y = 147, 147
x, y = 7, 7
depth = pixelDepth(quadtree, x, y)
if depth != -1:
    print(f"The pixel at ({x}, {y}) is at depth {depth} in the quadtree.")
else:
    print(f"The pixel at ({x}, {y}) was not found in the quadtree.")
    



# test searchSubspacesWithRange
x1, y1 = 170, 50
x2, y2 = 220, 140

new_quadtree = searchSubspacesWithRange(quadtree, x1, y1, x2, y2)


new_size = new_quadtree.size
display_image(new_quadtree, new_size)


# display_image(quadtree, height)
