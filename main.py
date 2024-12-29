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

    uniform, pixels = is_uniform_color(image, x, y, size)
    
    if uniform:
        return Node(x, y, size, color=tuple(image[y, x]), pixels=pixels)
    else:
        half_size = size // 2
        top_left = build_quadtree(image, x, y, half_size)
        top_right = build_quadtree(image, x + half_size, y, half_size)
        bottom_left = build_quadtree(image, x, y + half_size, half_size)
        bottom_right = build_quadtree(image, x + half_size, y + half_size, half_size)

        return Node(x, y, size, top_left=top_left, top_right=top_right, bottom_left=bottom_left, bottom_right=bottom_right)






def display_and_save_from_quadtree(node, original_image_shape, save_path="reconstructed_image.png"):
    
    
    height, width, channels = original_image_shape

    reconstructed_image = np.zeros((height, width, channels), dtype=np.uint8)


    def fill_image(node):
        if node.is_leaf():
            for px, py, _ in node.pixels:
                reconstructed_image[py, px] = node.color
        else:
            if node.top_left:
                fill_image(node.top_left)
            if node.top_right:
                fill_image(node.top_right)
            if node.bottom_left:
                fill_image(node.bottom_left)
            if node.bottom_right:
                fill_image(node.bottom_right)
    
    
    fill_image(node)
    
    
    save_image(reconstructed_image, save_path)


    plt.figure(figsize=(8, 8))
    plt.imshow(reconstructed_image)
    plt.axis('off')
    plt.title("Reconstructed Image from Quadtree")
    plt.show()





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




def compress_from_quadtree(node, target_size):
    compressed_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    scale_factor = node.size // target_size

    def fill_image_from_node(node):
        if node.is_leaf():
            
            start_x = node.x // scale_factor
            start_y = node.y // scale_factor
            size = node.size // scale_factor

            
            for i in range(start_y, start_y + size):
                for j in range(start_x, start_x + size):
                    compressed_image[i, j] = node.color
        else:
            
            if node.top_left:
                fill_image_from_node(node.top_left)
            if node.top_right:
                fill_image_from_node(node.top_right)
            if node.bottom_left:
                fill_image_from_node(node.bottom_left)
            if node.bottom_right:
                fill_image_from_node(node.bottom_right)

    fill_image_from_node(node)
    return compressed_image






def TreeDepth(node):
    if node.is_leaf():
        return 1
    else:
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




def pixelDepth(node, x, y, current_depth=1):
    
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
image = load_image("two-2-2.png")

height, width, _ = image.shape

if height == width:
    quadtree = build_quadtree(image, 0, 0, height)

# Calculate and print tree depth
depth = TreeDepth(quadtree)
print(f"The depth of the quadtree is: {depth}")


# compress size of image
target_size_for_compress = 8
compressed_image_from_tree = compress_from_quadtree(quadtree, target_size_for_compress)
save_image(compressed_image_from_tree, "compressed_image_from_tree.png")



x, y = 4, 5
depth = pixelDepth(quadtree, x, y)
print(f"The pixel at ({x}, {y}) is at depth {depth}")

# Example usage:
rect_top_left = (3, 2)  # x, y coordinates of the top-left corner
rect_bottom_right = (200, 100)  # x, y coordinates of the bottom-right corner

searchSubspacesWithRange(quadtree, rect_top_left, rect_bottom_right, image.shape)



# display_and_save_from_quadtree(quadtree, image.shape, save_path="reconstructed_image.png")
