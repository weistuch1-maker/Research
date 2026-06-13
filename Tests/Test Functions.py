import random


def run_dissection_tree_test(num_objects=10000, num_dimensions=10, tree_depth=4):
    """
    Generates high-dimensional data, builds a spatial dissection tree with
    heterogeneous axes per level, and compiles a comprehensive attribute report.
    """

    # 1. Generate a large amount of objects in many dimensions (values 0.0 to 1.0)
    generated_objects = [
        [random.random() for _ in range(num_dimensions)]
        for _ in range(num_objects)
    ]

    # 2. Generate a valid layout array for the axis tree
    # For a symmetrical tree of a given depth, the number of internal nodes is (2^depth) - 1
    num_internal_nodes = (2 ** tree_depth) - 1
    # Randomly assign a valid axis (0 to num_dimensions - 1) to every single internal node
    axis_tree_list = [random.randint(0, num_dimensions - 1) for _ in range(num_internal_nodes)]

    # 3. Generate the tree according to our spatial partition logic
    root_node = build_spatial_dissection_tree(generated_objects, axis_tree_list)

    # Helper functions to gather structural diagnostics for the report
    def get_max_depth(node):
        if node.is_leaf:
            return 0
        return 1 + max(get_max_depth(node.left), get_max_depth(node.right))

    def count_leaves(node):
        if node.is_leaf:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)

    # 4. Compile the attributes of the generated tree
    report = {
        "Dataset Metrics": {
            "Total Objects Generated": num_objects,
            "Dimensionality (N)": num_dimensions
        },
        "Tree Architecture": {
            "Target Depth": tree_depth,
            "Actual Maximum Depth": get_max_depth(root_node),
            "Total Leaf Nodes": count_leaves(root_node),
            "Root Partition Axis": root_node.axis,
            "Root Probabilities": f"Left: {root_node.p_left:.4f}, Right: {root_node.p_right:.4f}"
        },
        "Data Integrity Check": {
            "Sum of All Leaves (Root Cached Value)": root_node.sub_leaf_sum,
            "Matches Initial Object Count": root_node.sub_leaf_sum == num_objects
        }
    }

    # Generate a formatted text block for quick reading
    text_output = "==================================================\n"
    text_output += "        SPATIAL DISSECTION TREE TEST REPORT       \n"
    text_output += "==================================================\n"
    for section, metrics in report.items():
        text_output += f"\n## {section}\n"
        for key, val in metrics.items():
            text_output += f" - {key}: {val}\n"

    return root_node, text_output