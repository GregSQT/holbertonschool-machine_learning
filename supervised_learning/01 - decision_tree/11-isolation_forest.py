#!/usr/bin/env python3
"""
Exercice 11. IRF 2 : isolation random forests
"""

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Defines the Isolation Random Forest class.

    This class provides methods for training a forest of Isolation Random
    Trees, making predictions, and identifying suspicious instances based on
    mean depth.

    Attributes:
        numpy_predicts (list): List of functions for predicting with each tree.
        target (None): Placeholder for the target variable.
        numpy_preds (None): Placeholder for the predictions.
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        seed (int): Seed for the random number generator.
    """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initialize the Isolation Random Forest model.

        Args:
            n_trees (int, optional): Number of trees in the forest.
                Defaults to 100.
            max_depth (int, optional): Maximum depth of each tree.
                Defaults to 10.
            min_pop (int, optional): Minimum population of a node to not become
                a leaf. Defaults to 1.
            seed (int, optional): Seed for the random number generator.
                Defaults to 0.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Make predictions using the Isolation Random Forest model.

        Args:
            explanatory (numpy.ndarray): Explanatory features for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return predictions.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fit the Isolation Random Forest model to the training data.

        Args:
            explanatory (numpy.ndarray): Explanatory features for training.
            n_trees (int, optional): Number of trees to train. Defaults to 100.
            verbose (int, optional): Verbosity level. Defaults to 0.
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth, seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Identifies suspicious instances based on mean depth.

        Args:
            explanatory (numpy.ndarray): Explanatory features.
            n_suspects (int): Number of suspicious instances to identify.

        Returns:
            np.ndarray: Indices of the n_suspects rows in the explanatory
                array with the smallest mean depth.
        """
        depths = self.predict(explanatory)
        sorted_indices = np.argsort(depths)
        suspects = sorted_indices[:n_suspects]
        return explanatory[suspects], depths[suspects]
