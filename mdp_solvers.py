import time
import numpy as np 
from numpy import linalg as LA

def get_Q_value(mdp, state, action, V_states):
    """
    Input: 
    mdp: ToDoList MDP
    state: current state (tasks, time)
    action: index of action in mdp's tasks
    V_states: dictionary mapping states to current best (value, action)
    Output:
    total: Q-value of state
    """
    Q_value = 0
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
        Q_value += prob * (mdp.getReward(state, action, next_state) + mdp.getGamma() * next_state_value)
    return Q_value

def getValueAndAction(mdp, state, V_states):
    """
    Input: 
    mdp: ToDoList MDP
    state: current state (tasks, time)
    V_states: dictionary mapping states to current best (value, action)
    Output:
    best_value: value of state state
    best_action: index of action that yields highest value at current state
    """
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


def backward_induction(mdp, printTime=False):
    """
    Given a ToDoListMDP, perform value iteration/backward induction to find the optimal policy
    Input: ToDoListMDP
    Output: Optimal policy (and time elapsed if specified)
    """
    start = time.time()

    V_states = {} # maps state to (value, action)
    linearized_states = mdp.getLinearizedStates()
    # print "linearized states:", linearized_states
    numTasks = len(linearized_states)
    for state in linearized_states:
        V_states[state] = (0, None)

    # Perform Backward Iteration (Value Iteration 1 Time)
    for state in linearized_states:
        V_states[state] = getValueAndAction(mdp, state, V_states)        

    optimal_policy = {}
    for state in V_states:
        optimal_policy[state] = V_states[state][1]

    end = time.time()
    time_elapsed = end - start

    # mdp.calculatePseudorewards(V_states)
    
    if printTime:
        return optimal_policy, time_elapsed
    else:
        return optimal_policy


def value_iteration(mdp, printTime=False):
    """
    Given a ToDoListMDP, perform value iteration to find the optimal policy
    Input: ToDoListMDP
    Output: Optimal policy (and time elapsed if specified)
    """
    start = time.time()

    numTasks = len(mdp.getTasksList())
    V_states = {}
    for state in mdp.getStates():
        V_states[state] = (0, None)
    
    # perform value iteration with s iterations
    converged = False
    iterations = 0

    # Perform Value Iteration
    while not converged:
        print 'iteration', iterations
        iterations += 1
        next_V_states = {} 
        converged = True
        for state in V_states:
            next_V_states[state] = getValueAndAction(mdp, state, V_states)
            old_state_value = V_states[state][0]
            new_state_value = next_V_states[state][0]
            if abs(old_state_value - new_state_value) > 0.1:
                converged = False
        V_states = next_V_states      

    optimal_policy = {}
    for state in V_states:
        optimal_policy[state] = V_states[state][1]

    end = time.time()
    time_elapsed = end - start

    # mdp.calculatePseudorewards(V_states)
    
    if printTime:
        return optimal_policy, iterations, time_elapsed
    else:
        return optimal_policy


def policy_evaluation(mdp, policies, empty_A, empty_b):
    """
    given an MDP and a policy dictionary (from policy improvement)
    returns the V states for that policy for each state. V_states: {state: (V(s), action)}
    """

    states = mdp.getStates()
    gamma = mdp.getGamma()
    n = len(states)

    A = empty_A
    b = empty_b
    
    start = time.time()

    for i in range(n):

        state = states[i]
        action = policies[state]
        A[i][i] = -1

        for pair in mdp.getTransitionStatesAndProbs(state, action):
            
            next_state, prob = pair
            reward = mdp.getReward(state, action, next_state)
            j = states.index(next_state)

            A[i][j] = gamma * prob
            b[i] = b[i] - prob * reward


    end = time.time()

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
        best_action = getValueAndAction(mdp, state, V_states)[1]
        policies[state] = best_action

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










