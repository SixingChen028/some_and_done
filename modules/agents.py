import numpy as np


class agent_random:
    """
    An agent with random policy.
    """

    def __init__(self, prms, n_test):
        """
        Initialize the agent.
        """

        self.n_branch = prms['n_branch']
        self.n_test = n_test
        
        self.running_sum = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)
        self.running_count = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)


    def reset_buffer(self):
        """
        Reset running sum.
        """

        self.running_sum = np.zeros((self.n_test, self.n_branch))
        self.running_count = np.zeros((self.n_test, self.n_branch))


    def rollout(self, step):
        """
        Do a rollout.

        Returns:
            actions: an np.array with shape (n_test,)
        """

        if step == 0:
            actions = np.random.choice(self.n_branch, self.n_test)
        else:
            actions = np.argmin(self.running_count, axis = 1) # pick the actions that are least sampled
            
        return actions


    def update_buffer(self, actions, rewards):
        """
        Update buffer.
        """

        self.running_sum += np.eye(self.n_branch, dtype = int)[actions] * rewards.reshape((self.n_test, 1))
        self.running_count += np.eye(self.n_branch, dtype = int)[actions]


    def compute_q(self):
        """
        Compute Q values.
        """
        running_count = self.running_count.copy()
        running_count[running_count == 0] = 1

        Q = self.running_sum / running_count
        
        return Q


    def make_choices(self):
        """
        Make choices.

        Returns:
            choices: an np.array with shape (n_test,)
        """
        Q = self.compute_q()
        choices = np.argmax(Q, axis = 1) # (n_test,)
        
        return choices


class agent_max:
    """
    An agent that picks the action with the highest sampled reward (not average).
    """

    def __init__(self, prms, n_test):
        self.n_branch = prms['n_branch']
        self.n_test = n_test

        self.running_sum = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)
        self.running_count = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)
        self.max_reward = np.full((n_test, self.n_branch), -np.inf)  # initialize to -inf


    def reset_buffer(self):
        """
        Reset buffer.
        """

        self.running_sum = np.zeros((self.n_test, self.n_branch))
        self.running_count = np.zeros((self.n_test, self.n_branch))
        self.max_reward = np.full((self.n_test, self.n_branch), -np.inf)


    def rollout(self, step):
        """
        Do a rollout.

        Returns:
            actions: an np.array with shape (n_test,)
        """

        if step == 0:
            actions = np.random.choice(self.n_branch, self.n_test)
        else:
            actions = np.argmin(self.running_count, axis = 1) # pick the actions that are least sampled
            
        return actions
    

    def update_buffer(self, actions, rewards):
        """
        Update buffer.
        """

        self.running_sum += np.eye(self.n_branch, dtype = int)[actions] * rewards.reshape((self.n_test, 1))
        self.running_count += np.eye(self.n_branch, dtype = int)[actions]

        row_indices = np.arange(self.n_test)
        # Use np.maximum to do in-place elementwise comparison and update
        self.max_reward[row_indices, actions] = np.maximum(
            self.max_reward[row_indices, actions], rewards
        )

        # for i in range(self.n_test):
        #     a = actions[i]
        #     self.max_reward[i, a] = max(self.max_reward[i, a], rewards[i])


    def make_choices(self):
        """
        Make choices.

        Returns:
            choices: an np.array with shape (n_test,)
        """

        choices = np.argmax(self.max_reward, axis = 1)
        return choices


# class agent_softmax:
#     """
#     An agent with random policy.
#     """

#     def __init__(self, prms, n_test):
#         """
#         Initialize the agent.
#         """

#         self.n_branch = prms['n_branch']
#         self.n_test = n_test
        
#         self.running_sum = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)
#         self.running_count = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)


#     def reset_buffer(self):
#         """
#         Reset running sum.
#         """

#         self.running_sum = np.zeros((self.n_test, self.n_branch))
#         self.running_count = np.zeros((self.n_test, self.n_branch))


#     def rollout(self, step):
#         """
#         Do a rollout.

#         Returns:
#             actions: an np.array with shape (n_test,)
#         """

#         if step == 0:
#             actions = np.random.choice(self.n_branch, self.n_test)
#         else:
#             actions = np.argmin(self.running_count, axis = 1) # pick the actions that are least sampled
            
#         return actions


#     def update_buffer(self, actions, rewards):
#         """
#         Update buffer.
#         """

#         self.running_sum += np.eye(self.n_branch, dtype = int)[actions] * rewards.reshape((self.n_test, 1))
#         self.running_count += np.eye(self.n_branch, dtype = int)[actions]


#     def compute_q(self):
#         """
#         Compute Q values.
#         """
#         running_count = self.running_count.copy()
#         running_count[running_count == 0] = 1

#         Q = self.running_sum / running_count
        
#         return Q

    
#     def make_choices(self, temperature = 1.0):
#         """
#         Make choices using softmax sampling.

#         Returns:
#             choices: an np.array with shape (n_test,)
#         """

#         Q = self.compute_q() # shape: (n_test, n_branch)

#         # apply softmax with temperature
#         Q_scaled = Q / temperature
#         exp_Q = np.exp(Q_scaled - np.max(Q_scaled, axis = 1, keepdims = True))  # for numerical stability
#         probs = exp_Q / np.sum(exp_Q, axis = 1, keepdims = True) # shape: (n_test, n_branch)

#         # sample actions from the softmax probabilities
#         choices = np.array([
#             np.random.choice(self.n_branch, p=probs[i])
#             for i in range(self.n_test)
#         ])

#         return choices


# class agent_UCB:
#     def __init__(self, prms, n_test):
#         self.n_branch = prms['n_branch']
#         self.n_test = n_test
        
#         self.running_sum = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)
#         self.running_count = np.zeros((self.n_test, self.n_branch)) # (n_test, n_branch)

#     def reset_buffer(self):
#         self.running_sum = np.zeros((self.n_test, self.n_branch))
#         self.running_count = np.zeros((self.n_test, self.n_branch))

#     def rollout(self, step):
#         if step == 0:
#             actions = np.random.choice(self.n_branch, self.n_test)
#         else:
#             actions = np.argmax(UCB, axis = 1)
            
#         return actions

#     def update_buffer(self, actions, rewards):
#         self.running_sum += np.eye(self.n_branch, dtype = int)[actions] * rewards.reshape((self.n_test, 1))
#         self.running_count += np.eye(self.n_branch, dtype = int)[actions]

#     def compute_q(self):
#         running_count_replaced = self.running_count.copy()
#         running_count_replaced[running_count_replaced == 0] = 1

#         Q = self.running_sum / running_count_replaced
        
#         return Q

#     def compute_ucb(self, step, c = 1):
#         Q = self.compute_q()
#         UCB = Q + c * np.sqrt(np.log(step + 1) / (self.running_count + 1e-5))
#         return UCB

#     def make_choices(self):
#         Q = self.compute_q()
#         choices = np.argmax(Q, axis = 1) # (n_test,)
        
#         return choices



# class agent_gaussian:
#     def __init__(self, prms, n_test):
#         self.n_branch = prms['n_branch']
#         self.n_test = n_test
        
#         self.precision_init = 1e-10
#         self.precision_sample = 1.5

#         self.mean = np.zeros((self.n_test, self.n_branch))
#         self.precision = np.ones((self.n_test, self.n_branch)) * self.precision_init
        
#     def reset_buffer(self):
#         self.mean = np.zeros((self.n_test, self.n_branch))
#         self.precision = np.ones((self.n_test, self.n_branch)) * self.precision_init

#     def rollout(self, step):
#         if step == 0:
#             actions = np.random.choice(self.n_branch, self.n_test)
#         else:
#             actions = np.argmax(self.mean + self.precision ** (-2), axis = 1)
            
#         return actions

#     def update_buffer(self, actions, rewards):
#         precision_likelihood = np.eye(self.n_branch, dtype = int)[actions] * self.precision_sample
        
#         self.mean = (precision_likelihood * rewards[:, None] + self.precision * self.mean) / (precision_likelihood + self.precision)
#         self.precision = self.precision + precision_likelihood

#     def make_choices(self):
#         choices = np.argmax(self.mean, axis = 1) # (n_test,)

#         return choices



