#!/usr/bin/env python3
"""
Exercise 2. Let's print our Tree
"""
import numpy as np


class Node:
    """ Defines a node for a decision tree """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Class constructor for Node instances.
        """
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
        Class constructor for Node instances.
        """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Calculate the number of nodes in the tree.
        """
        if self.is_leaf:
            return 1

        lcount = self.left_child.count_nodes_below(only_leaves=only_leaves)
        rcount = self.right_child.count_nodes_below(only_leaves=only_leaves)
        return lcount + rcount + (not only_leaves)

    def left_child_add_prefix(self, text):
        """ Adds the prefix in the line for tree display """
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """ Adds the prefix in the line for tree display """
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       "+x)+"\n"
        return (new_text.rstrip())

    def __str__(self):
        """ Defines the printing format for a node """
        if self.is_root:
            t = "root"
        else:
            t = "-> node"
        return f"{t} [feature={self.feature}, threshold={self.threshold}]\n"\
            + self.left_child_add_prefix(str(self.left_child))\
            + self.right_child_add_prefix(str(self.right_child))

    def get_leaves_below(self):
        """ Returns the list of leaves below the current Node instance """
        left_leaves = self.left_child.get_leaves_below()
        right_leaves = self.right_child.get_leaves_below()
        return left_leaves + right_leaves


class Leaf(Node):
    """
    Defines the class Leaf
    """

    def __init__(self, value, depth=None):
        """
        Class constructor for Leaf instances
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns the depth of the leaf """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        return 1

    def __str__(self):
        """ Defines the printing format for a Leaf instance """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """ Returns the current Leaf instance in a list """
        return [self]


class Decision_Tree():
    """ Defines the decision tree """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Class constructor for Decision_tree instances
        """
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
        """ Returns the max depth of the decision tree """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns the number of nodes is the tree
        If only_leaves is True, excludes the root and internal nodes
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ Defines the printing format for a Decision_Tree instance """
        return self.root.__str__()+"\n"

    def get_leaves(self):
        """ Returns the list of all the leaves in the decision tree """
        return self.root.get_leaves_below()
