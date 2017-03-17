import time
import numpy as np 
from numpy import linalg as LA

def get_Q_value(mdp, state, action, V_states):
    total = 0
    trans_states_and_probs = mdp.getTransitionStatesAndProbs(state, action)
    for pair in trans_states_and_probs:
        next_state = pair[0]
        tasks = next_state[0]
        time = next_state[1]
        prob = pair[1]
        # IMPORTANT: below varies on val iter or policy iter
        v = V_states[next_state]
        if isinstance(v, tuple):
            next_state_value = V_states[next_state][0]
        else:
            next_state_value = V_states[next_state]
        total += prob * (mdp.getReward(state, action, next_state) + mdp.getGamma() * next_state_value)
    return total

def choose_action(mdp, state, V_states):
    possible_actions = mdp.getPossibleActions(state)   
    best_action = None
    best_value = -float('inf')
    if mdp.isTerminal(state):
        best_value = 0
        best_action = 0
    for a in possible_actions:
        q_value = get_Q_value(mdp, state, a, V_states)
        if q_value > best_value:
            best_value = q_value
            best_action = a
    return (best_value, best_action)

def policy_evaluation(mdp, policies, empty_A, empty_b):
    """
    given an MDP and a policy dictionary (from policy improvement)
    returns the V states for that policy for each state. V_states: {state: (V(s), action)}
    """
    # print(policies)
    # start = time.time()

    states = mdp.getStates()
    gamma = mdp.getGamma()
    n = len(states)
    print n

    """
    A = []
    b = [0 for i in range(n)]
    # print 'n', n
    for i in range(n):
        state = states[i]
        action = policies[state]
        row = [0 for index in range(n)]
        row[i] = -1
        # add a helper function in mdp class to compute neighbors and their indices (for future)
        for pair in mdp.getTransitionStatesAndProbs(state, action):
            (next_state, prob) = pair
            j = states.index(next_state)
            reward = mdp.getReward(state, action, next_state)

            row[j] = mdp.getGamma() * prob
            b[i] = b[i] - prob * reward  
        A.append(row)

    A = np.array(A)
    b = np.array(b)

    print A
    """

    # """
    A = empty_A
    b = empty_b
    # print 'n', n
    
    start = time.time()

    for i in range(n):

        # start = time.time()

        state = states[i]
        action = policies[state]
        A[i][i] = -1
        # print mdp.getTransitionStatesAndProbs(state, action)
        # if i == n-1: print 'this state', state
        for pair in mdp.getTransitionStatesAndProbs(state, action):
            
            next_state, prob = pair
            reward = mdp.getReward(state, action, next_state)
            j = states.index(next_state)

            # if i == n-1: 
                # print 'next state', next_state
                # print 'prob', prob
                # print 'reward', reward

            A[i][j] = gamma * prob
            b[i] = b[i] - prob * reward



        # ((state_j, prob_j), (state_k, prob_k)) = mdp.getTransitionStatesAndProbs(state, action)
        
        # reward_j = mdp.getReward(state, action, state_j)
        # reward_k = mdp.getReward(state, action, state_k)
        
        # start = time.time()
        # j = states.index(state_j)
        # end = time.time()
        # # if i == 0: print 'searching for index', end - start
        # start = time.time()
        # k = states.index(state_k)
        # end = time.time()
        # if i == 0: print 'searching for index', end - start

        # FAST STUFF
        # A[i][j] = gamma * prob_j
        # A[i][k] = gamma * prob_k
        # A[i][i] = -1
        # b[i] = - prob_j * reward_j - prob_k * reward_k

        # row = np.array([-1 if index==i else gamma*prob_j if index==j else gamma*prob_k if index==k else 0 for index in range(n)])
        # start = time.time()
        # if A is not None:
        #     A = np.vstack((A, row))
        # else: 
        #     A = np.array([row])
        # end = time.time()
        # if i % 500 == 0: print 'time for stack', end - start
        # b = np.append(b, - prob_j * reward_j - prob_k * reward_k)

        # end = time.time()
        # if i > n - 10: print 'time for iteration', end - start

    end = time.time()
    # print 'creating matrix time', end - start 
    # print A
    # print b
    # """


    start = time.time()
    v = LA.solve(A, b)
    end = time.time()
    # print 'solving matrix time', end - start

    v_states = {state: value for (state, value) in zip(states, v)}
    
    # end = time.time()
    # print 'policy evaluation time', end - start

    return v_states

    
def policy_extraction(mdp, V_states):
    """
    given an MDP and V_states (from policy evaluation)
    returns the optimal policy (policy is dictionary{states: action index})
    """
    # start = time.time()
    # for every state, pick the action corresponding to the highest Q-value
    policies = {}
    states = mdp.getStates()
    for state in states:
        best_action = choose_action(mdp, state, V_states)[1]
        policies[state] = best_action

    # end = time.time()
    # print 'policy extraction time', end - start

    return policies

def policy_iteration(mdp):
    """
    given an MDP
    performs policy iteration and returns the converged policy
    """

    states = mdp.getStates()
    policy = {}
    new_policy = {}
    # create initial policies
    for state in states:
        # new_policy[state] = 0
        tasks = state[0]
        # set initial policy of each state to the first possible action (index of first 0)
        if 0 in tasks:
            new_policy[state] = tasks.index(0)
        else:
            new_policy[state] = 0
    
    start = time.time()
    n = len(states)
    empty_A = np.array([np.array([0 for j in range(n)]) for i in range(n)])
    empty_b = np.array([0 for i in range(n)])

    iterations = 0
    # repeat until policy converges
    while policy != new_policy:
        print 'iterations', iterations
        iterations += 1

        policy = new_policy
        v_states = policy_evaluation(mdp, policy, empty_A, empty_b)
        new_policy = policy_extraction(mdp, v_states)
        
    end = time.time() 

    start_state = mdp.getStartState()
    state = start_state
    optimal_tasks = []
    while not mdp.isTerminal(state):
        optimal_action = policy[state]
        task = mdp.getTasksList()[optimal_action]
        next_state_tasks = list(state[0])[:]
        next_state_tasks[optimal_action] = 1
        next_state = (tuple(next_state_tasks), state[1] + task.getTimeCost())
        state = next_state
        optimal_tasks.append(task)

    optimal_policy = [task.getDescription() for task in optimal_tasks]
    time_elapsed = end - start

    return optimal_policy, iterations, time_elapsed











