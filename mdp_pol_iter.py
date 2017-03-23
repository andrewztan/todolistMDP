import time
import numpy as np 
from numpy import linalg as LA
from scipy.sparse import linalg as sLA
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

np.set_printoptions(threshold='nan')

def val_iter(mdp, policy):
    """
    Given a ToDoListMDP and a policy, perform value iteration to find values from the policy
    Input: MDP, policy
    Output: Values according to policy
    """

    V_states = {}
    for state in mdp.getStates():
        V_states[state] = 0

    # perform value iteration with s iterations
    converged = False
    iterations = 0
    # print V_states

    # Perform Value Iteration
    while not converged:
        # print 'iteration', iterations
        iterations += 1
        next_V_states = {} 
        converged = True
        for state in V_states:
            action = policy[state]
            next_V_states[state] = do_action(mdp, state, action, V_states)

            old_state_value = V_states[state]
            new_state_value = next_V_states[state]
            if abs(old_state_value - new_state_value) > 10.0:
                converged = False
        V_states = next_V_states

    return V_states

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

def do_action(mdp, state, action, V_states):
    total = 0
    trans_states_and_probs = mdp.getTransitionStatesAndProbs(state, action)
    for pair in trans_states_and_probs:
        next_state, prob = pair
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

def policy_evaluation(mdp, policy):
    start = time.time()
    """
    given an MDP and a policy dictionary (from policy improvement)
    returns the V states for that policy for each state. V_states: {state: (V(s), action)}
    """
    # print(policies)
    # start = time.time()

    # states = mdp.getStates()
    # gamma = mdp.getGamma()
    # n = len(states)
    # print n

    """ 
    fast = False
    fast = True

    if not fast:
        # print 'not fast'
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
        # print A

    else: 
        start = time.time()
        A = np.zeros((n, n))
        b = np.zeros(n)
        # end = time.time()
        # print 'copy time', end - start

        # start = time.time()

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

                j = mdp.getStateIndex(next_state)

                A[i][j] = gamma * prob
                b[i] = b[i] - prob * reward
            
        end = time.time()
        print 'creating matrix time', end - start 
        # print A

    # start = time.time()
    # A = csc_matrix(A)
    # end = time.time()
    # print 'convert matrix A', end - start

    start = time.time()
    v = sLA.spsolve(csc_matrix(A), b)
    # print v
    end = time.time()
    print 'solving matrix time', end - start

    v_states = {state: value for (state, value) in zip(states, v)}
    """

    v_states = val_iter(mdp, policy)
    end = time.time()
    print 'pol eval', end - start

    return v_states

    
def policy_extraction(mdp, V_states):
    """
    given an MDP and V_states (from policy evaluation)
    returns the optimal policy (policy is dictionary{states: action index})
    """
    # start = time.time()
    # for every state, pick the action corresponding to the highest Q-value
    policy = {}
    states = mdp.getStates()
    for state in states:
        best_action = choose_action(mdp, state, V_states)[1]
        policy[state] = best_action

    # end = time.time()
    # print 'policy extraction time', end - start

    return policy

def policy_iteration(mdp):
    """
    given an MDP
    performs policy iteration and returns the converged policy
    """
    start = time.time()
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
    end = time.time()
    # print 'init policies', end - start
    
    n = len(states)

    start = time.time()
    empty_A = np.zeros((n, n))
    empty_b = np.zeros(n)
    end = time.time()
    # print 'init empty arrays', end - start

    start = time.time()
    iterations = 0
    # repeat until policy converges
    while policy != new_policy:
        print 'iteration', iterations
        iterations += 1

        policy = new_policy
        v_states = policy_evaluation(mdp, policy)
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











