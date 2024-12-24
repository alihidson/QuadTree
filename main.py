import numpy as np
from PIL import Image

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

# Loading image
image = load_image("one-color.png")

height, width, _ = image.shape

if height == width:
    quadtree = build_quadtree(image, 0, 0, height)

# Calculate and print tree depth
depth = TreeDepth(quadtree)
print(f"The depth of the quadtree is: {depth}")
