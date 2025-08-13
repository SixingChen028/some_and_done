import numpy as np


def get_prms(
        n_branch = 2,
        n_depth = 4,
        sig_r = 0.05
    ):
    """
    Generate a dictionary with parameters of a given world.

    Args:
        n_branch: an integer. branching facter
        n_depth: an integer: tree depth
        sig_r: a float. observation noise std of reward at a single node
    
    Returns:
        prms: a dictionary of parameters
            n_branch: an integer. branching facter
            n_depth: an integer: tree depth
            sig_g: a float. observation noise std for path value at a leaf node
            sig_b: a float. mean reward std at a single node
            n_leaves: an integer. number of leaf nodes
            lpa: an integer. number of leaf nodes per action (branch)

    """
    
    n_leaves = int(n_branch ** n_depth)
    lpa = int(n_leaves / n_branch)
    sig_b = 1 # np.sqrt(1 / n_depth) # np.sqrt(1 / (n_depth + 1))
    sig_g = sig_r * np.sqrt(n_depth) # accumulate observation noise over depths
    
    prms = {
        'n_branch': n_branch,
        'n_depth': n_depth,
        'sig_g': sig_g,
        'sig_b': sig_b,
        'n_leaves': n_leaves,
        'lpa': lpa,
    }

    return prms


def node_from_transitions(
        transitions,
        prms
    ):
    """
    Return index of final state from a set of transitions

    Args:
        transitions: an np.array with shape (n_batch, n_depth)
            each row shows the transition sequence, e.g., [0, 1, 0, 1]
            a transition sequence can be understood as binary code, e.g., 5 for [0, 1, 0, 1]
        
    Returns:
        node: an np. array with shape (n_batch,)
            each element indicate the leaf node after the transition sequence
    """
    
    # compute numbers of nodes in each layer and list them from leaf layer to root layer
    # e.g. [8, 4, 2, 1] when n_branch = 2 and n_depth = 4
    branch_factor = prms['n_branch'] ** np.arange(prms['n_depth'])[::-1]

    # do point-wise dot between transitions and branch factors to get the final state (transition from binary)
    node = np.sum(transitions * branch_factor, axis = -1).astype(int) # point-wise product
    
    return node


def assign_mus(
        prms,
        mus = [],
        prev_mus = [],
        transitions = [],
        all_mus = [],
        return_all = False,
        iter = 0
    ):
    """
    Assign reward means in leaf node (or in the whole tree)

    Args:
        prms: a dictionary of parameters
        mus: start from an empty list
        prev_mus: start from an empty list
        transitions: start from an empty list
        all_mus: start from an empty list
        return_all: a boolean variable
            if True, return all means instead of only leaves
        iter: an integer
    
    Returns:
        mus: an np.array with a shape of (n_leaves,)
            if return_all is True, return a list of np.arrays where each element has a shape of (n_depth,)
            each element corresponds to the means along the path to a leaf node
    """
    
    # initialize the first step of the recursion
    if iter == 0:
        mus = np.zeros(prms['n_leaves']) # initialize means for leaves
        all_mus = [] # initialize means for all nodes

    # iterate over child branches
    for i in range(prms['n_branch']):

        # concatenate the child index to the transitions
        new_transitions = np.concatenate([transitions, [i]])

        # sample a reward mean and concatenate to means
        new_mus = np.concatenate([prev_mus, [np.random.normal() * prms['sig_b']]])

        # if not reach the leaf node
        if len(new_mus) < prms['n_depth']:

            # go one step deeper
            mus = assign_mus(prms, mus, new_mus, new_transitions, all_mus, return_all, iter + 1)

            # return all means
            if return_all:
                all_mus = mus

        # if reach the leaf node
        else:

            # get the lead node based on transitions
            node = node_from_transitions(new_transitions, prms)

            # store the means of the path corresponding to a leaf node
            if return_all:
                all_mus.append(new_mus)

            # sum the means over the path corresponding to a leaf node
            else:
                mus[node] = np.sum(new_mus) # only store value of leaf
                # the value of the leaf is the sum of all parent values
    
    # return all means
    if return_all:
        return all_mus # n_leaves x n_depth
    
    return mus


def sample_rewards(
        actions,
        mus,
        prms,
        max_depth = None,
        p = None,
        all_mus = None
    ):
    """
    Given a set of actions of any shape, return rewards drawn from the model

    Args:
        actions: an np.array with arbitary shape, e.g., (n_samples, n_repetitions)
        mus: an np.array with shape (n_leaves,)
        parms: a dictionary of parameters
        max_depth: None or an integer. max depth considered
            max depth is set to the max depth of the tree if not specified
        p: None or an np.array with shape (n_branch,). transition probability at each branch
            p is set to uniform if not specified
        all_mus: None or a list of all path reward means

    p is the transition probability at each step (None gives uniform transition probabilities)
    """

    # initialize transitions to the shape of actions, then add a dimension of depth
    # transitions has shape of (..., n_depth)
    transitions = np.zeros(actions.shape + (prms['n_depth'],)) # final axis is depth axis
    
    # set the depth-1 transitions fo actions
    transitions[..., 0] = actions

    # sample random transitions from depth-2
    for i in range(prms['n_depth'] - 1):
        transitions[..., i + 1] = np.random.choice(np.arange(prms['n_branch']), actions.shape, p = p)

    # get the leaf nodes with transitions
    final_node = node_from_transitions(transitions, prms)
    
    # use max depth of the tree
    if max_depth is None:
        sample_mus = mus
    
    # use partial depth of the tree
    else:
        sample_mus = np.sum(np.array(all_mus)[:, :max_depth], axis = 1)

    # add observation noise to rewards
    rewards = np.random.normal(0, prms['sig_g'], actions.shape) + sample_mus[final_node]

    return rewards


def get_true_action_rewards(
        mus,
        prms,
        p = None
    ):
    """
    Get true rewards for depth-1 actions

    Args:
        mus: an np.array with a shape of (n_leaves,)
        prms: a parameter dictioinary
        p: a float. transition probability at each node
    
    Returns
        rewards: an np.array with shape of (n_branch,). true action values
    """

    # if transition probabilities are uniform
    if p is None:
        # slice leaf nodes according to which depth-1 branch it belongs to, then average means
        return np.array([np.mean(mus[(a * prms['lpa']):((a + 1) * prms['lpa'])]) for a in range(prms['n_branch'])]) 
    
    # if transition probabilities are non-uniform
    else:
        # initialize depth-1 rewards
        rewards = np.zeros(prms['n_branch'])

        # loop throught leaves
        for node in range(prms['n_leaves']):

            # back-track a leaf node back to a transition sequence
            # transitions is an n_branch-base string, e.g., 00010 for node 2
            transitions = np.base_repr(node, base = prms['n_branch']).zfill(prms['n_depth'])
            transitions = [int(transitions[i]) for i in range(prms['n_depth'])]

            # compute the probabilities of all the transitions
            # the first transition is deterministic so p is set to 1
            ps = np.concatenate([[1.0], np.array(p)[np.array(transitions)[1:]]])

            # compute the probability of the whole transition sequence
            # prob is the probability of transiting to the leaf node from the root
            prob = np.prod(ps)

            # add the reward of the path corresponding to the leaf node to the depth-1 action
            rewards[transitions[0]] += prob * mus[node]
        
        return rewards


def eval(
        n_depth,
        n_branch = 2,
        n_eval = 500,
        n_world = 2000,
        sig_r = 0.1,
    ):
    """
    Evaluate rewards in different environments.

    Args:
        n_depth: an integer
        n_branch: an integer
        n_eval: an integer. number of evaluations per environment
        n_world: an integer. number of environments
    
    Rreturns:
        all_rewards: an np.array with shape (n_world, n_branch, n_eval). all sampled rewards
        all_groups: an np.array with shape (n_world,). all relative true rewards between two actions
    """

    all_rewards = []
    all_groups = []

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

    # loop through environments
    for _ in range(n_world):

        # draw a world
        mus = assign_mus(prms)

        # get actions
        actions = np.arange(prms['n_branch'])[:, np.newaxis] * np.ones((1, n_eval)) # (n_branch, n_eval)

        # sample rewards
        rewards = sample_rewards(actions, mus, prms) # (n_branch, n_eval)

        # get true rewards
        true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)

        # record
        all_rewards.append(rewards)
        all_groups.append(true_action_rewards[0] > true_action_rewards[1])
    
    all_rewards = np.array(all_rewards) # (n_world, n_branch, n_eval)
    all_groups = np.array(all_groups) # (n_world,)

    return all_rewards, all_groups


def cum_log_LR(x, mu, sigma):
    """
    Compute cumulative log-likelihood ratio
    """

    mu1, mu2 = mu[0], mu[1]
    log_LR = - 1 / (2 * sigma ** 2) * (2 * x * (mu2 - mu1) + mu1 ** 2 - mu2 ** 2)
    
    return np.cumsum(log_LR)


def get_prob(p_main, n_branch):
    """
    Get the probability distribution
    """

    p_other = (1 - p_main) / (n_branch - 1)
    return np.append(p_main, np.full(n_branch - 1, p_other))