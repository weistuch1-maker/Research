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