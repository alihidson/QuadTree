import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y, size, color=None, pixels=None, top_left=None, top_right=None, bottom_left=None, bottom_right=None):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.pixels = pixels
        
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
    
    def is_leaf(self):
        return self.top_left is None and self.top_right is None and self.bottom_left is None and self.bottom_right is None




def load_image(file_path):
    image = Image.open(file_path).convert('RGB')
    return np.array(image)




def save_image(image_array, file_name):
    image = Image.fromarray(image_array)
    image.save(file_name)





def is_uniform_color(image, x, y, size):
    color = image[y, x]
    pixels = []
    for i in range(y, y + size):
        for j in range(x, x + size):
            if not np.array_equal(image[i, j], color):
                return False, None
            pixels.append((i, j, tuple(image[i, j])))  # Storing coordinates and color
    return True, pixels




def build_quadtree(image, x, y, size):
    if size == 1:
        return Node(x, y, size, color=tuple(image[y, x]), pixels=[(x, y, tuple(image[y, x]))])


    # Divide the image region into 4 parts
    half_size = size // 2
    top_left = (x, y)
    top_right = (x + half_size, y)
    bottom_left = (x, y + half_size)
    bottom_right = (x + half_size, y + half_size)

    # Create empty node (the root node or internal node without color)
    node = Node(x, y, size)
    
    # Check if the region is uniform in color for each part
    uniform_top_left, pixels_top_left = is_uniform_color(image, *top_left, half_size)
    uniform_top_right, pixels_top_right = is_uniform_color(image, *top_right, half_size)
    uniform_bottom_left, pixels_bottom_left = is_uniform_color(image, *bottom_left, half_size)
    uniform_bottom_right, pixels_bottom_right = is_uniform_color(image, *bottom_right, half_size)
    
    
    
    if uniform_top_left:
        node.top_left = Node(*top_left, half_size, color=tuple(image[top_left[1], top_left[0]]), pixels=pixels_top_left)
    else:
        node.top_left = build_quadtree(image, *top_left, half_size)

    if uniform_top_right:
        node.top_right = Node(*top_right, half_size, color=tuple(image[top_right[1], top_right[0]]), pixels=pixels_top_right)
    else:
        node.top_right = build_quadtree(image, *top_right, half_size)

    if uniform_bottom_left:
        node.bottom_left = Node(*bottom_left, half_size, color=tuple(image[bottom_left[1], bottom_left[0]]), pixels=pixels_bottom_left)
    else:
        node.bottom_left = build_quadtree(image, *bottom_left, half_size)

    if uniform_bottom_right:
        node.bottom_right = Node(*bottom_right, half_size, color=tuple(image[bottom_right[1], bottom_right[0]]), pixels=pixels_bottom_right)
    else:
        node.bottom_right = build_quadtree(image, *bottom_right, half_size)

    return node





def display_image(node, target_size):
    
    if node.size < target_size:
        raise ValueError(f"Target size {target_size} is larger than the root node size {node.size}.")

    # Initialize the image array with the target size
    image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    def fill_image_preorder(node):
        if node.is_leaf():
            # Calculate where to place the color in the image
            start_x = node.x
            start_y = node.y
            size = node.size

            # Place the node color in the image at the specified position
            for i in range(start_y, start_y + size):
                for j in range(start_x, start_x + size):
                    image[i, j] = node.color
        else:
            # Pre-order traversal: Process root first, then children
            if node.top_left:
                fill_image_preorder(node.top_left)
            if node.top_right:
                fill_image_preorder(node.top_right)
            if node.bottom_left:
                fill_image_preorder(node.bottom_left)
            if node.bottom_right:
                fill_image_preorder(node.bottom_right)

    # Begin the pre-order traversal to fill the image
    fill_image_preorder(node)

    # Display the image using matplotlib
    plt.imshow(image)
    plt.axis('off')  # Hide axis
    plt.show()

    # return image











def searchSubspacesWithRange(node, rect_top_left, rect_bottom_right, original_image_shape):
    
    height, width, channels = original_image_shape
    output_image = np.full((height, width, channels), fill_value=255, dtype=np.uint8)  # White background

    def overlaps(node, rect_top_left, rect_bottom_right):
        
        node_rect_top_left = (node.x, node.y)
        node_rect_bottom_right = (node.x + node.size - 1, node.y + node.size - 1)

        return not (
            node_rect_bottom_right[0] < rect_top_left[0] or
            node_rect_top_left[0] > rect_bottom_right[0] or
            node_rect_bottom_right[1] < rect_top_left[1] or
            node_rect_top_left[1] > rect_bottom_right[1]
        )

    def fill_output_image(node):
        
        if node.is_leaf():
            if overlaps(node, rect_top_left, rect_bottom_right):
                for px, py, _ in node.pixels:
                    if rect_top_left[0] <= px <= rect_bottom_right[0] and rect_top_left[1] <= py <= rect_bottom_right[1]:
                        output_image[py, px] = node.color
        else:
            if node.top_left and overlaps(node.top_left, rect_top_left, rect_bottom_right):
                fill_output_image(node.top_left)
            if node.top_right and overlaps(node.top_right, rect_top_left, rect_bottom_right):
                fill_output_image(node.top_right)
            if node.bottom_left and overlaps(node.bottom_left, rect_top_left, rect_bottom_right):
                fill_output_image(node.bottom_left)
            if node.bottom_right and overlaps(node.bottom_right, rect_top_left, rect_bottom_right):
                fill_output_image(node.bottom_right)

    fill_output_image(node)

    # Display and save the resulting image
    save_image(output_image, "subspaces_in_range.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.title("Subspaces in Range")
    plt.show()



def compress_image(node, target_size):
    
    if node.size < target_size:
        raise ValueError(f"Target size {target_size} is larger than the root node size {node.size}.")

    # Ensure the size is divisible
    if node.size % target_size != 0:
        raise ValueError(f"Root node size {node.size} must be divisible by target size {target_size}.")

    # Initialize the compressed image array
    compressed_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # Calculate the scale factor
    scale_factor = node.size // target_size

    def fill_image_preorder(node):
        if node.is_leaf():
            # Calculate where to place the color in the compressed image
            start_x = node.x // scale_factor
            start_y = node.y // scale_factor
            size = node.size // scale_factor

            # Fill the corresponding area in the compressed image
            for i in range(start_y, start_y + size):
                for j in range(start_x, start_x + size):
                    compressed_image[i, j] = node.color
        else:
            # Pre-order traversal: Process root first, then children
            if node.top_left:
                fill_image_preorder(node.top_left)
            if node.top_right:
                fill_image_preorder(node.top_right)
            if node.bottom_left:
                fill_image_preorder(node.bottom_left)
            if node.bottom_right:
                fill_image_preorder(node.bottom_right)

    # Begin the pre-order traversal to fill the compressed image
    fill_image_preorder(node)

    return compressed_image






def TreeDepth(node):
    if node.is_leaf():
        return 0  # Leaf nodes are at level 0 from their perspective.

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
    
    if node.is_leaf():
        
        for px, py, _ in node.pixels:
            if px == x and py == y:
                return current_depth
        return -1
    
    
    half_size = node.size // 2
    
    if x < node.x + half_size and y < node.y + half_size:
        return pixelDepth(node.top_left, x, y, current_depth + 1)
    elif x >= node.x + half_size and y < node.y + half_size:
        return pixelDepth(node.top_right, x, y, current_depth + 1)
    elif x < node.x + half_size and y >= node.y + half_size:
        return pixelDepth(node.bottom_left, x, y, current_depth + 1)
    elif x >= node.x + half_size and y >= node.y + half_size:
        return pixelDepth(node.bottom_right, x, y, current_depth + 1)

    return -1 



# Loading image
image = load_image("hidson.png")


height, width, _ = image.shape

if height == width:
    quadtree = build_quadtree(image, 0, 0, height)

# Calculate and print tree depth
depth = TreeDepth(quadtree)
print(f"The depth of the quadtree is: {depth}")


# Set the target size
target_size_for_compress = 8

# Check and compress the image
try:
    compressed_image = compress_image(quadtree, target_size_for_compress)
    save_image(compressed_image, "compressed_image_from_tree_preorder.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(compressed_image)
    plt.axis('off')
    plt.title("Compressed Image (Pre-Order Traversal)")
    plt.show()
except ValueError as e:
    print(f"Error: {e}")

x, y = 4, 5
depth = pixelDepth(quadtree, x, y)
print(f"The pixel at ({x}, {y}) is at depth {depth}")

# Example usage:
rect_top_left = (3, 2)  # x, y coordinates of the top-left corner
rect_bottom_right = (100, 50)  # x, y coordinates of the bottom-right corner

# searchSubspacesWithRange(quadtree, rect_top_left, rect_bottom_right, image.shape)



display_image(quadtree, height)
