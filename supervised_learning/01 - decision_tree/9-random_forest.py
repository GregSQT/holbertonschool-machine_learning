#!/usr/bin/env python3
"""
Exercise 9. Random forests
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

    def update_bounds_below(self):
        """ Calculate the number of nodes in the tree. """
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
        """ Updates indicators"""
        def is_large_enough(A):
            """ Check if large enough """
            return np.all(
                np.array([np.greater(A[:, feature], self.lower[feature])
                          for feature in self.lower]), axis=0)

        def is_small_enough(A):
            """ Check if small enough """
            return np.all(
                np.array([np.less_equal(A[:, feature], self.upper[feature])
                          for feature in self.upper]), axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, A):
        """
        Returns the recursively computed prediction outcome
        for the current node
        """
        if A[self.feature] > self.threshold:
            return self.left_child.pred(A)
        else:
            return self.right_child.pred(A)


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
        """ Calculate the number of nodes in the tree. """
        return 1

    def __str__(self):
        """ Defines the printing format for a Leaf instance """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """ Returns the current Leaf instance in a list """
        return [self]

    def update_bounds_below(self):
        """ Does nothing since there are no nodes below a leaf """
        pass

    def pred(self, x):
        """ Returns the predicted value for the current Leaf instance """
        return self.value


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

    def update_bounds(self):
        """ Updates the lower and upper bounds of the decision tree """
        self.root.update_bounds_below()

    def pred(self, A):
        """ Returns the prediction value of the decision tree """
        return self.root.pred(A)

    def update_predict(self):
        """ Faster way of computing the prediction value of the decision tree,
        using the indicator function with the known bounds of the leaves. """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value for leaf in leaves]),
            axis=0
        )

    def fit(self, explanatory, target, verbose=0):
        """
        Updates the split_criterion attribute with the splitting method to be
        used and assigns the sub_population attribute with a 1D np.array of
        booleans of size target.size (number of individuals in the the training
        set).
        The i-th value of this array is True if and only if the i-th individual
        visits the node (so for the root, all the values are True).
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion  # < --- to be
            # defined later
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(
                only_leaves=True)}
    - Accuracy on training data : {self.accuracy(
                    self.explanatory, self.target)}""")  # < --- to be
            # defined later

    def np_extrema(self, arr):
        """ Returns the extreme values of an np.array. """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ Iteratively chooses splits from the root on depending on the
        split_criterion attribute. Completely random for now. """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """ Recursively fits a decision tree node by splitting the data based
        on the best criterion. """
        node.feature, node.threshold = self.split_criterion(node)

        max_criterion = self.explanatory[:, node.feature] > node.threshold

        left_population = node.sub_population & max_criterion
        right_population = node.sub_population & ~max_criterion

        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        np.unique(self.target[left_population]).size == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         np.unique(self.target[right_population]).size == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ Creates a leaf child node for the decision tree with the specified
        sub-population. """
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth+1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Creates a new node child for the decision tree with the specified
        sub-population. """
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """ Calculates the accuracy of the decision tree model on the provided
        test data. """
        return np.sum(
            np.equal(
                self.predict(test_explanatory), test_target))/test_target.size

    def possible_thresholds(self, node, feature):
        """ Computes possible thresholds for splitting the specified feature
        in the given node's sub-population. """
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:]+values[:-1])/2

    def Gini_split_criterion_one_feature(self, node, feature):
        """
        Computes the best threshold and corresponding average Gini impurity for
        splitting the specified feature in the given node, based on the Gini
        impurity criterion.

        Args:
            node (Node): The node to split.
            feature (int): The index of the feature to split on.

        Returns:
            tuple: A tuple containing the best threshold and its corresponding
                average Gini impurity.
        """
        # Compute possible thresholds
        thresholds = self.possible_thresholds(node, feature)

        # Get the indexes of individuals in the node's sub_population
        indices = np.arange(0, self.explanatory.shape[0])[node.sub_population]
        # Number of individuals in the sub_population
        num_individuals = indices.size

        # Get feature values for individuals in the node's sub_population
        feature_values = (self.explanatory[:, feature])[node.sub_population]

        # Compute masks for left and right child nodes
        filter_left = np.greater(
            feature_values[:, np.newaxis], thresholds[np.newaxis, :])
        filter_right = ~filter_left

        # Only consider individuals in the node's sub_population
        target_reduced = self.target[indices]

        # Get unique classes in the node's sub_population
        classes = np.unique(self.target)

        # Compute class masks for left and right children
        classes_mask = np.equal(target_reduced[:, np.newaxis], classes)

        left_class_mask = np.logical_and(
            classes_mask[:, :, np.newaxis], filter_left[:, np.newaxis, :])
        right_class_mask = np.logical_and(
            classes_mask[:, :, np.newaxis], filter_right[:, np.newaxis, :])

        # Compute Gini impurities for left and right children
        gini_left = 1 - np.sum(
            np.square(np.sum(left_class_mask, axis=0)),
            axis=0) / np.sum(filter_left, axis=0) / num_individuals
        gini_right = 1 - np.sum(
            np.square(np.sum(right_class_mask, axis=0)),
            axis=0) / np.sum(filter_right, axis=0) / num_individuals

        # Sum average of Gini impurities
        gini_sum = gini_left + gini_right

        # Find index of threshold with smallest total impurity
        min_index = np.argmin(gini_sum)
        min_threshold = thresholds[min_index]

        return np.array([min_threshold, gini_sum[min_index]])

    def Gini_split_criterion(self, node):
        """ Computes the best feature and threshold for splitting the node
        based on the Gini impurity criterion. """
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]


class Random_Forest():
    """
    Defines a Random Forest ensemble of decision trees.

    Attributes:
        numpy_predicts (list): List to store predictions from individual trees.
        target (numpy.ndarray): The target variable.
        numpy_preds (np.ndarray): Predictions generated by the Random Forest.
        n_trees (int): The number of decision trees in the ensemble.
        max_depth (int): The maximum depth of each decision tree.
        min_pop (int): The minimum number of samples required to split a node.
        seed (int): The seed value for random number generation.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Constructor method for a Random_Forest instance. """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Generates predictions for a given set of explanatory variables.

        Args:
            explanatory (numpy.ndarray): The explanatory variables.

        Returns:
            numpy.ndarray: Predictions generated by the Random Forest ensemble.
        """
        predictions = np.array([tree_predict(explanatory)
                               for tree_predict in self.numpy_preds])

        mode = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions)

        return mode

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Fits the Random Forest model to the training data.

        Args:
            explanatory(numpy.ndarray): The explanatory variables.
            target(numpy.ndarray): The target variable.
            n_trees(int, optional): The number of decision trees in the
                ensemble. Defaults to 100.
            verbose(int, optional): Verbosity mode(0 or 1). Defaults to 0.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop, seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                                      self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Computes the accuracy of the Random Forest model on test data.

        Args:
            test_explanatory (np.ndarray): The explanatory variables of the td.
            test_target (np.ndarray): The target variable of the test data.

        Returns:
            float: The accuracy of the Random Forest model on the test data.
        """
        return np.sum(
            np.equal(
                self.predict(test_explanatory), test_target))/test_target.size
