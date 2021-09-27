"""
Student Name: Oindril Dutta
GT User ID: odutta3
GT ID: 903459041
"""

import numpy as np

class DTLearner(object):
    """
    This is a Decision Tree Learner. It is a recursive algorithm that builds a
    tree based on the data provided.

    :param leaf_size: The maximum number of samples to be aggregated at a leaf,
    defaults to 1.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information
    for debugging. If verbose = False your code should not generate ANY output.
    When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.verbose = verbose
        self.leaf_size = leaf_size
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "odutta3"

    def build_tree(self, data_x, data_y):
        """
        Builds the decision tree based on data provided regardless of existing
        tree

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # get the rows (# of samples) of the input data
        rows = data_x.shape[0]
        # get the fallback leaf to return if unable to continue the tree
        leaf = np.array([-1, np.mean(data_y), np.nan, np.nan])
        # if the number of rows is less than the leaf size, or if there's no
        # variety in the output data, aka all inputs lead to the same output,
        # then result to the fallback
        if rows <= self.leaf_size or len(np.unique(data_y)) == 1:
            return leaf
        # get all the pearson correlation coefficients, an indication of the
        # best column to split on
        correlation_coefficients_per_x_column = np.abs(
            np.corrcoef(
                data_x,
                y=data_y,
                rowvar=False,
            ))[:-1, -1]
        try:
            # determine best feature i to split on by getting the index with
            # the biggest correlation coefficient
            best_feature_i = np.nanargmax(
                correlation_coefficients_per_x_column
            )
        except ValueError:
            return leaf
        # get the value to split the input data on by getting the median of the
        # best feature i column
        split_val_i = np.median(data_x[:, best_feature_i])
        # get an array of true and false indices based on the split val to know
        # what input data to use for each child tree
        left_child_index = data_x[:, best_feature_i] <= split_val_i
        right_child_index = data_x[:, best_feature_i] > split_val_i
        # if all the booleans are true or false, then the best coefficient
        # cannot split it, thus we should return the fallback leaf, otherwise
        # it will cause infinite recursion
        if np.all(left_child_index) or np.all(right_child_index):
            return leaf
        # build child trees using recursion guided by the boolean array indexes
        left_child_tree = self.build_tree(
            data_x[left_child_index, :],
            data_y[left_child_index],
        )
        right_child_tree = self.build_tree(
            data_x[right_child_index, :],
            data_y[right_child_index],
        )
        # if the left child is a leaf fallback, then return 2, since if the
        # left is a leaf then the right child node has to be 2 steps away, else
        # return the left child shape (steps away from right node)
        right_tree_distance = left_child_tree.shape[
            0] + 1 if left_child_tree.ndim > 1 else 2
        # build the root node based on the best feature column index
        # (highest pearson correlation coefficient), the median value of that
        # column index, 1 (steps away from left node), and the root of the tree
        # on the right
        root = np.array([
            best_feature_i,
            split_val_i,
            1,
            right_tree_distance,
        ])
        # stack the arrays together recursively to generate the 2d
        # representation of a decision tree
        return np.vstack((
            root,
            left_child_tree,
            right_child_tree,
        ))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner by either adding a new tree or merging	
        with existing tree

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        # build the new tree based on the data
        new_tree = self.build_tree(data_x, data_y)
        # if a tree doesn't exist yet, use the new tree
        if self.tree is None:
            self.tree = new_tree
        # else if there is a tree, then simply stack the old on top of the new
        else:
            self.tree = np.vstack((
                self.tree,
                new_tree,
            ))
        # in the rare case there's only one array in the tree, hence a 1D array
        # with shape (4), then expand it's dimensions to a 2D array with shape
        # (4, 1) so it can be vertically stacked later on as evidence is added
        if len(self.tree.shape) == 1:
            self.tree = np.expand_dims(self.tree, axis=0)

    def search(self, point_array, node=0):
        """
        Recursively find the best y for a specific x-array with a starting node

        :param point_array: A specific query row.
        :type point_array: numpy.ndarray
        :param node: The index of the tree currently being searched.
        :type node: int
        :return: The predicted result of the point_array
        :rtype: numpy.ndarray
        """
        # for this node, get the best column index, and it's split value
        best_column_i, best_column_split_val = self.tree[node, 0:2]
        # if the index of the best feature column is -1, then it's the leaf,
        # so return the value at this leaf
        if best_column_i == -1:
            return best_column_split_val
        # else recursively search the tree, going left if the query point_array
        # value at the index is less than the split value at the current node
        # of the tree, and going right otherwise (2 is left, 3 is right)
        direction = 2 if point_array[
            int(best_column_i), ] <= best_column_split_val else 3
        # recursively search the tree, get the node to go down to by looking
        # at the number of indexes to go down by getting the count at either
        # the left or right direction of the node
        return self.search(
            point_array,
            node + int(self.tree[node, direction]),
        )

    def query(self, points):
        """
        Estimate a set of test points given the model we built.

        :param points: A numpy array with each row corresponding to a specific
        query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the
        trained model
        :rtype: numpy.ndarray
        """
        # map over all point_arrays to query and call a search for each one,
        # returning an np array of results
        return np.array([self.search(point_array) for point_array in points])


if __name__ == "__main__":
    pass
