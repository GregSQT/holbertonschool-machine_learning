#!/usr/bin/env python3
"""
Exercise 0-Depth of a decision tree
"""
import numpy as np


class Node:
    """
    Defines the classe Node
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
        """
        def max_depth_recursive(node, depth):
            if self.is_leaf is True:
                return self.depth
            else:
                left_depth = self.left_child.max_depth_below()
                right_depth = self.right_child.max_depth_below()
                return max(left_depth, right_depth)

        def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the tree.
        """
        if self.is_leaf:
            return 1

        lcount = self.left_child.count_nodes_below(only_leaves=only_leaves)
        rcount = self.right_child.count_nodes_below(only_leaves=only_leaves)
        return lcount + rcount + (not only_leaves)


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
        return self.depth

    def count_nodes_below(self, only_leaves=False) :
    return 1


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
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False) :
        return self.root.count_nodes_below(only_leaves=only_leaves)
