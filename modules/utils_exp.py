import numpy as np

def parse_raw_to_trials(raw, num_trial):
    """
    Parse raw data to trials.
    """

    trials = [[] for _ in range(num_trial)]
    main_flag = False
    trial_index = 0

    # loop through events
    for event in raw:

        # if in the main exp
        if main_flag:

            # update trial index
            if 'trial_index' in event.keys():
                trial_index = event['trial_index']

            # log event to trial
            trials[trial_index].append(event)
        
        # enter the main exp
        if event['event'] == 'timeline.start.main':
            main_flag = True

    return trials


def parse_trials_to_data(trials):
    """
    Parse trials to data.
    """

    data = {
        'graphs': [],
        'rewards': [],
        'starts': [],
        'time_limits': [],
        'hover_seqs': [],
        'visit_seqs': [],
        'hover_time_seqs': [],
        'visit_time_seqs': [],
    }

    # loop through trials
    for trial in trials:
        # loop through events in the trial
        for event in trial:

            # new trial
            if 'graph' in event.keys():
                # initialize trial recording
                hover_seq = []
                visit_seq = []
                hover_time_seq = []
                visit_time_seq = []

                # log trial info
                data['graphs'].append(event['graph'])
                data['rewards'].append(event['rewards'])
                data['starts'].append(event['start'])
                data['time_limits'].append(event['time_limit'])
            
            # imagination
            if event['event'] == 'graph.imagine':
                hover_seq.append(event['state'])
                hover_time_seq.append(event['time'])
            
            # navigation
            if event['event'] == 'graph.visit':
                visit_seq.append(event['state'])
                visit_time_seq.append(event['time'])
        
        # log trial info
        data['hover_seqs'].append(hover_seq)
        data['visit_seqs'].append(visit_seq)
        data['hover_time_seqs'].append(hover_time_seq)
        data['visit_time_seqs'].append(visit_time_seq)
    
    return data


def max_depth(adj_list, root):
    """
    Get max depth of a tree.
    """

    def dfs(node, depth):
        # return current depth the node has no children
        if not adj_list[node]:
            return depth
        
        max_child_depth = depth
        for child in adj_list[node]:
            max_child_depth = max(max_child_depth, dfs(child, depth + 1))

        return max_child_depth

    return dfs(root, 0)


def list_to_dict(adj_list):
    """
    Transform adjacency list to child dict.
    """

    child_dict = {}

    for parent, children in enumerate(adj_list):
        # add entries for nodes that have children
        if children:
            child_dict[parent] = children

    return child_dict


def merge_adjacent(lst):
    """
    Merge identical adjacent elements in a list.
    """

    # check if the list is empty
    if not lst:
        return []

    # start with the first element
    merged_list = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            merged_list.append(lst[i])

    return merged_list


def is_node_in_tree(node, child_dict):
    """
    Check if a node is in a tree.
    """

    if node in child_dict:
        return True
    
    for children in child_dict.values():
        if node in children:
            return True
    
    return False


def is_node_depth_1(node, child_dict, start):
    """
    Check if a node is a depth-1 node.
    """

    depth_1_nodes = child_dict[start]

    return node in depth_1_nodes


def get_depth(node, child_dict, start, depth = 0):
    """
    Get the depth of a node in a tree.
    """

    # if the start node is the target node, return the depth
    if start == node:
        return depth
    
    # if the start node is not in the tree, return -1 (node not found)
    if start not in child_dict:
        return -1
    
    # recursively search for the target node in the children
    for child in child_dict[start]:
        result = get_depth(node, child_dict, child, depth + 1)
        if result != -1:
            return result
    
    # if the node is not found in any subtree
    return -1


def get_action_value(node, child_dict, rewards):
    """
    Get the action value of a node in a tree following a random policy.
    """

    # if the node has no children, return its reward
    if node not in child_dict:
        return rewards[node]
    
    # if the node has children, recursively compute the action values of the children
    child_nodes = child_dict[node]
    child_values = [get_action_value(child, child_dict, rewards) for child in child_nodes]
    mean_child_value = sum(child_values) / len(child_values)

    # compute action value
    action_value = rewards[node] + mean_child_value
    
    # return the expected value
    return action_value


def relationship(child_dict, node1, node2):
    """
    A function for testing the relationship of two nodes in a tree.
    """

    if node1 == node2:
        return 'self'
    elif node1 in child_dict.keys() and node2 in child_dict[node1]:
        return 'child'
    elif node2 in child_dict.keys() and node1 in child_dict[node2]:
        return 'parent'
    else:
        for parent, children in child_dict.items():
            if node1 in children and node2 in children:
                return 'sibling'
    return 'others'


def segment_lengths(lst, start_element):
    segments = []
    counts = []
    current_segment = []
    
    for num in lst:
        if num == start_element:
            if current_segment:
                segments.append(current_segment)
                counts.append(len(current_segment) - 1)  # Exclude start element
            current_segment = [num]
        else:
            current_segment.append(num)
    
    if current_segment:
        segments.append(current_segment)
        counts.append(len(current_segment) - 1)  # Exclude start element
    
    return counts