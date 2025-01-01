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
        if self.top_left == None and self.top_right == None and self.bottom_left == None and self.bottom_right == None:
            return True
        else:
            return False