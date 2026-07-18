import random
import matplotlib.pyplot as plt
import numpy as np


class DissectionNode:
    def __init__(self, axis=None, p_left=0.5, p_right=0.5, value=None, left=None, right=None, parent=None):
        self.axis = axis
        self.p_left = p_left
        self.p_right = p_right
        self.value = value
        self.parent = parent

        # Initialize internal private child attributes
        self._left = None
        self._right = None

        # Use properties to handle parent linking automatically
        self.left = left
        self.right = right

        self.sub_leaf_sum = 0
        self.update_sum()

    @property
    def is_leaf(self):
        return self._left is None and self._right is None

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, child):
        self._left = child
        if child is not None:
            child.parent = self
        self.update_sum()

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, child):
        self._right = child
        if child is not None:
            child.parent = self
        self.update_sum()

    def update_sum(self):
        """Calculates the sum of immediate children without walking the whole tree."""
        if self.is_leaf:
            self.sub_leaf_sum = self.value if self.value is not None else 0
        else:
            left_sum = self._left.sub_leaf_sum if self._left else 0
            right_sum = self._right.sub_leaf_sum if self._right else 0
            self.sub_leaf_sum = left_sum + right_sum

    def update_leaf_value(self, new_value):
        """
        Updates a leaf's value and bubbles the changes up to the root.
        """
        if not self.is_leaf:
            raise ValueError("Cannot directly assign a value to a non-leaf partitioning node.")

        self.value = new_value

        # Bubble up the changes to all ancestors
        curr = self
        while curr is not None:
            curr.update_sum()
            curr = curr.parent


def build_spatial_dissection_tree(objects, axis_tree_list, bounds=None, node_index=0):
    """
    Builds a symmetrical N-dimensional dissection tree where nodes at the same depth
    can split on entirely different axes.

    :param objects: List of N-dimensional points (floats between 0 and 1)
    :param axis_tree_list: Flat list representing a complete binary tree of split axes
                           arranged in breadth-first order.
    :param bounds: Internal parameter tracking the current spatial boundaries
    :param node_index: Internal parameter tracking the current node's position in the axis list
    """
    if bounds is None:
        if not objects:
            raise ValueError("Cannot automatically determine dimensionality without objects.")
        num_dimensions = len(objects[0])
        bounds = [(0.0, 1.0) for _ in range(num_dimensions)]

    # BASE CASE: If the index falls outside the axis tree list,
    # it means this position is designated as a leaf node.
    if node_index >= len(axis_tree_list) or axis_tree_list[node_index] is None:
        return DissectionNode(value=len(objects))

    # 1. Fetch the unique axis assigned to this specific node
    current_axis = axis_tree_list[node_index]
    axis_min, axis_max = bounds[current_axis]
    split_point = (axis_min + axis_max) / 2.0

    # 2. Dissect the space
    left_objects = [obj for obj in objects if obj[current_axis] <= split_point]
    right_objects = [obj for obj in objects if obj[current_axis] > split_point]

    # 3. Calculate density probabilities
    total_objects = len(objects)
    if total_objects > 0:
        p_left = len(left_objects) / total_objects
        p_right = len(right_objects) / total_objects
    else:
        p_left, p_right = 0.5, 0.5

    # 4. Narrow down spatial bounds for children
    left_bounds = list(bounds)
    left_bounds[current_axis] = (axis_min, split_point)

    right_bounds = list(bounds)
    right_bounds[current_axis] = (split_point, axis_max)

    # 5. Calculate array indices for the left and right children
    left_child_index = 2 * node_index + 1
    right_child_index = 2 * node_index + 2

    # 6. Recursively build sub-trees
    left_child = build_spatial_dissection_tree(left_objects, axis_tree_list, left_bounds, left_child_index)
    right_child = build_spatial_dissection_tree(right_objects, axis_tree_list, right_bounds, right_child_index)

    return DissectionNode(
        axis=current_axis,
        p_left=p_left,
        p_right=p_right,
        left=left_child,
        right=right_child
    )


def trace_point_path(root, point, bounds=None):
    """
    Traverses the dissection tree to find the leaf containing the point.
    Returns a list of (success_count, failure_count) tuples for each split level.

    :param root: The root DissectionNode of the tree
    :param point: List or tuple of floats representing the N-dimensional point
    :param bounds: Optional initial spatial boundaries. Defaults to [0.0, 1.0] per dimension.
    """
    if bounds is None:
        # Default space configuration to match the tree builder
        bounds = [(0.0, 1.0) for _ in range(len(point))]

    # Create a local mutable copy of the boundaries to narrow down during traversal
    current_bounds = list(bounds)
    path_counts = []
    current_node = root

    # Loop down until we hit a terminated leaf node
    while not current_node.is_leaf:
        axis = current_node.axis
        axis_min, axis_max = current_bounds[axis]
        split_point = (axis_min + axis_max) / 2.0

        # Determine which branch contains our target point
        if point[axis] <= split_point:
            success_child = current_node.left
            failure_child = current_node.right
            # Update bounds for subsequent lower levels
            current_bounds[axis] = (axis_min, split_point)
        else:
            success_child = current_node.right
            failure_child = current_node.left
            # Update bounds for subsequent lower levels
            current_bounds[axis] = (split_point, axis_max)

        # Extract the aggregated object counts from both sub-trees
        success_count = success_child.sub_leaf_sum if success_child else 0
        failure_count = failure_child.sub_leaf_sum if failure_child else 0

        # Append the pair to our history log
        path_counts.append((success_count, failure_count))

        # Move down to the next node in the path
        current_node = success_child

    return path_counts


def weighted_density(tree_root_list, weights, point):
    """
    Using pre-calculated trees, determine the weighted density in a single point
    :param tree_root_list: DissectionNode objects representing roots of trees
    :param weights: Ordered weight of each tree in density calculation (adds to 1)
    :param point: N-dimensional point
    """
    if len(tree_root_list) != len(weights):
        raise ValueError("Tree list length mismatches with weights list")

    weighted_prob = 0
    # Loop through all trees, adding weighted probabilities
    for i in range(len(tree_root_list)):
        tree_prob = 1
        beta_dist_variables = trace_point_path(tree_root_list[i], point)

        for success, failure in beta_dist_variables:
            alpha = success + 1
            beta = success + 1
            p = random.betavariate(alpha=alpha, beta=beta)
            tree_prob *= p

        weighted_prob += tree_prob * weights[i]

    return weighted_prob


def sample_point_from_tree(root, a0, b0, num_dimensions, bounds=None):
    """
    Generates an N-dimensional point by stochastically traversing the tree
    using Beta-distributed branch probabilities, then sampling uniformly within the leaf bounds.

    :param root: The root DissectionNode of the tree
    :param a0: Beta distribution prior for the left branch (must be > 0)
    :param b0: Beta distribution prior for the right branch (must be > 0)
    :param num_dimensions: Total number of dimensions (N) of the space
    :param bounds: Optional initial spatial boundaries. Defaults to [0.0, 1.0] per dimension.
    """
    if bounds is None:
        # Initialize the global hyper-cube bounds if not provided
        current_bounds = [(0.0, 1.0) for _ in range(num_dimensions)]
    else:
        current_bounds = list(bounds)

    current_node = root

    # 1. Stochastically descend the tree until hitting a leaf node
    while not current_node.is_leaf:
        axis = current_node.axis
        axis_min, axis_max = current_bounds[axis]
        split_point = (axis_min + axis_max) / 2.0

        # Gather the object counts from the left and right subtrees
        left_count = current_node.left.sub_leaf_sum if current_node.left else 0
        right_count = current_node.right.sub_leaf_sum if current_node.right else 0

        # 2. Setup Beta distribution parameters using counts + user-defined priors
        alpha = left_count + a0
        beta = right_count + b0

        # Sample the probability of choosing the left branch
        p_left = random.betavariate(alpha=alpha, beta=beta)

        # 3. Choose a child branch at random based on p_left
        if random.random() < p_left:
            # Step Left (Success branch for this choice)
            current_bounds[axis] = (axis_min, split_point)
            current_node = current_node.left
        else:
            # Step Right (Failure branch for this choice)
            current_bounds[axis] = (split_point, axis_max)
            current_node = current_node.right

    # 4. Leaf reached: Uniformly sample a point within the final N-dimensional sub-cube bounds
    generated_point = [
        random.uniform(dim_min, dim_max)
        for dim_min, dim_max in current_bounds
    ]

    return generated_point


def one_dim_tree(object_amount, result_amount, a0, b0):

    n_vec = [[random.random()] for _ in range(object_amount)]
    root = build_spatial_dissection_tree(n_vec, [0])

    p_list = []
    for i in range(result_amount):
        new_p = sample_point_from_tree(root, a0, b0, num_dimensions=1)
        p_list.append(new_p)

    bins_amount = 20
    counts, bins = np.histogram(p_list, bins=bins_amount, range=(0, 1))
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()


def two_dim_tree(object_amount, result_amount, a0, b0):

    n_vec = [[random.random(), random.random()] for _ in range(object_amount)]
    root = build_spatial_dissection_tree(n_vec, [0, 1, 1])

    x_vals = []
    y_vals = []
    for i in range(result_amount):
        new_p = sample_point_from_tree(root, a0, b0, num_dimensions=2)
        x_vals.append(new_p[0])
        y_vals.append(new_p[1])

    n_vec_x_vals = []
    n_vec_y_vals = []
    for i in n_vec:
        n_vec_x_vals.append(i[0])
        n_vec_y_vals.append(i[1])

    plt.scatter(x_vals, y_vals)
    plt.scatter(n_vec_x_vals, n_vec_y_vals)
    plt.show()


def main():

    object_amount = 1
    result_amount = 2000

    a0, b0 = 1, 1

    two_dim_tree(object_amount, result_amount, a0, b0)
    #one_dim_tree(10, result_amount, a0, b0)


main()

