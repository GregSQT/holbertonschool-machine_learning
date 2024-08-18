#!/usr/bin/env python3
""" Isolation Random Tree: Implementation of an isolation random tree for
isolation forest algorithm. """

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Isolation Random Tree class for constructing an isolation random tree.

    Attributes:
        rng (numpy.random.Generator): Random number generator.
        root (Node): Root node of the tree.
        explanatory (numpy.ndarray): Explanatory features.
        max_depth (int): Maximum depth of the tree.
        predict (function): Function for predicting the target values.
        min_pop (int): Minimum population for a node to be considered for
            splitting.
    """

    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes an Isolation Random Tree.

        Args:
            max_depth (int, optional): Maximum depth of the tree.
                Defaults to 10.
            seed (int, optional): Seed for the random number generator.
                Defaults to 0.
            root (Node, optional): Root node of the tree. Defaults to None.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """ Defines the printing format for a Node instance. """
        return self.root.__str__

    def depth(self):
        """ Computes the depth of an isolation random tree. """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the number of nodes in the ensemble.

        Args:
            only_leaves (bool, optional): Defines if the root and internal
                nodes are excluded to count only the leaves. Defaults to False.
        """
        return self.root.count_nodes_below(only_leaves)

    def update_bounds(self):
        """ Updates the bounds for nodes in the tree. """
        return self.root.update_bounds_below()

    def get_leaves(self):
        """ Gets all leaf nodes in the tree. """
        return self.root.get_leaves_below()

    def update_predict(self):
        """ Updates the predict function based on the tree. """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def np_extrema(self, arr):
        """
        Calculates the minimum and maximum values in an array.

        Args:
            arr (numpy.ndarray): Input array.

        Returns:
            tuple: Minimum and maximum values.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Generates a random split criterion for a given node.

        Args:
            node (Node): Node for which to generate the split criterion.

        Returns:
            tuple: Feature index and threshold for splitting.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf child node.

        Args:
            node (Node): Parent node.
            sub_population (numpy.ndarray): Subpopulation indices.

        Returns:
            Leaf: Leaf child node.
        """
        leaf_child = Leaf(node.depth + 1)
        leaf_child.depth = node.depth+1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates an internal node child.

        Args:
            node (Node): Parent node.
            sub_population (numpy.ndarray): Subpopulation indices.

        Returns:
            Node: Internal node child.
        """
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Fits a node in the tree.

        Args:
            node (Node): Node to fit.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        max_criterion = self.explanatory[:, node.feature] > node.threshold

        left_population = node.sub_population & max_criterion
        right_population = node.sub_population & ~max_criterion

        is_left_leaf = (node.depth == self.max_depth - 1)\
            or (np.sum(left_population) <= self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth == self.max_depth - 1)\
            or (np.sum(right_population) <= self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the isolation random tree to the data.

        Args:
            explanatory (numpy.ndarray): Explanatory features.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(
            explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}""")
