#!/usr/bin/env python3
""" Node, Leaf, and Decision_Tree classes """
import numpy as np


class Node:
    """ Defines a node of the decision tree. """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
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
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Computes the depth of a decision tree using recursion. """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the number of nodes in the tree.

        Args:
            only_leaves (bool, optional): Defines if the root and internal
                nodes are excluded to count only the leaves. Defaults to False.
        """
        if self.is_leaf:
            return 1

        lcount = self.left_child.count_nodes_below(only_leaves=only_leaves)
        rcount = self.right_child.count_nodes_below(only_leaves=only_leaves)
        return lcount + rcount + (not only_leaves)

    def left_child_add_prefix(self, text):
        """ Adds the prefix in the line for correct printing of the tree. """
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  "+x)+"\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """ Adds the prefix in the line for correct printing of the tree. """
        lines = text.split("\n")
        new_text = "    +--"+lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       "+x)+"\n"
        return (new_text.rstrip())

    def __str__(self):
        """ Defines the printing format for a Node instance. """
        if self.is_root:
            t = "root"
        else:
            t = "-> node"
        return f"{t} [feature={self.feature}, threshold={self.threshold}]\n"\
            + self.left_child_add_prefix(str(self.left_child))\
            + self.right_child_add_prefix(str(self.right_child))

    def get_leaves_below(self):
        """ Returns the list of leaves below the current Node instance. """
        left_leaves = self.left_child.get_leaves_below()
        right_leaves = self.right_child.get_leaves_below()
        return left_leaves + right_leaves

    def update_bounds_below(self):
        """ Updates the lower and upper bounds observed in the data subset
        associated with the Node instance. Attributes lower and upper are
        dictionnaries, the keys represent the features and each feature has a
        bound. """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        flag = "left"
        for child in [self.left_child, self.right_child]:
            child.upper = self.upper.copy()
            child.lower = self.lower.copy()
            feature, threshold = self.feature, self.threshold

            if flag:
                child.lower[feature] = max(
                    threshold, child.lower.get(feature, threshold))
            else:
                child.upper[feature] = min(
                    threshold, child.upper.get(feature, threshold))

            flag = None

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Computes the indicator function from the Node.lower and Node.upper
        dictionaries and stores it in an attribute Node.indicator.

        The indicator function takes in a 2D np.array A of shape
        (n_individuals, n_features). The output of the indicator is is a
        1D np.array of shape (n_indiveduals,), containing boolean values.
        The i-th element of this output array is set to True if the
        corresponding i-th individual meets the conditions specified by the
        node “n”; otherwise, it is set to False.
        """

        def is_large_enough(A):
            return np.all(
                np.array([np.greater(A[:, feature], self.lower[feature])
                          for feature in self.lower]), axis=0)

        def is_small_enough(A):
            return np.all(
                np.array([np.less_equal(A[:, feature], self.upper[feature])
                          for feature in self.upper]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


class Leaf(Node):
    """ Defines a leaf of the decision tree. A leaf has no childs. """

    def __init__(self, value, depth=None):
        """
        Class constructor for Leaf instances.

        Args:
            value (int): The value held by the leaf.
            depth (int, optional): The depth of the leaf. Defaults to None.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Returns the depth of the leaf. """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Number of nodes in the tree. Returns 1 since the leaf is the last
        node. """
        return 1

    def __str__(self):
        """ Defines the printing format for a Leaf instance. """
        return (f"-> leaf [value={self.value}] ")

    def get_leaves_below(self):
        """ Returns the current Leaf instance in a list. """
        return [self]

    def update_bounds_below(self):
        """ Does nothing since there are no nodes below a leaf. """
        pass


class Decision_Tree():
    """ Defines a decision tree. """

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Class constructor for Decision_tree instances.

        Args:
            max_depth (int, optional): _description_. Defaults to 10.
            min_pop (int, optional): _description_. Defaults to 1.
            seed (int, optional): _description_. Defaults to 0.
            split_criterion (str, optional): description. Defaults to "random".
            root (bool, optional): _description_. Defaults to None.
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
        """ Returns the max depth of the decision tree. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Returns the number of nodes is the tree. If only_leaves is True,
        excludes the root and internal nodes. """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """ Defines the printing format for a Decision_Tree instance. """
        return self.root.__str__()+"\n"

    def get_leaves(self):
        """ Returns the list of all the leaves in the decision tree. """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """ Updates the lower and upper bounds of the decision tree. """
        self.root.update_bounds_below()
