#!/usr/bin/env python3
"""
Exercise 0-Depth of a decision tree
"""
import numpy as np


class Node:
    """
        Class constructor for Node instances.

        Args:
            feature (int, optional): _description_. Defaults to None.
            threshold (float, optional): _description_. Defaults to None.
            left_child (Node, optional): _description_. Defaults to None.
            right_child (Node, optional): _description_. Defaults to None.
            is_root (bool, optional): _description_. Defaults to False.
            depth (int, optional): _description_. Defaults to 0.
        """
    def __init__(self, feature=None,
                 threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Recursively calculate the depth of a decision tree
        Input : self
        """
        def max_depth_recursive(node, depth):
            """
            Recursively dig into the tree to get the depth
            """
            if node.is_leaf is True:
                return depth
            else:
                left_depth = max_depth_recursive(node.left_child, depth + 1)
                right_depth = max_depth_recursive(node.right_child, depth + 1)
                return max(left_depth, right_depth)
        return max_depth_recursive(self, 0)


class Leaf(Node):
    """
    Defines the class Leaf
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf
        """
        return self.depth


class Decision_Tree():
    """
    Defines the class Decision-Tree
    """
    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the max depth of the tree
        """
        return self.root.max_depth_below()
