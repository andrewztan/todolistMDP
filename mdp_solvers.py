import time
import numpy as np

def value_iteration(mdp, gamma=1.0):
    """
    Given a ToDoListMDP, perform value iteration/backward induction to find the optimal policy
    Input: MDP, gamma decay
    Output: Optimal policy, number of iterations, empirical runtime
    """
    numTasks = len(mdp.getTasksList())
    V_states = {}
    for state in mdp.getStates():
        V_states[state] = (0, None)

    start = time.time()
    
    # perform value iteration with s iterations
    converged = False
    iterations = 0
    # print V_states
    while not converged:
        print('iteration', iterations)
        iterations += 1
        next_V_states = {} 
        converged = True
        for state in V_states:
            next_V_states[state] = choose_action(mdp, state, V_states)

            old_state_value = V_states[state][0]
            new_state_value = best_value
            if abs(old_state_value - new_state_value) > 0.1:
                converged = False
        V_states = next_V_states

    end = time.time()

    start_state = mdp.getStartState()
    state = start_state
    optimal_tasks = []
    while not mdp.isTerminal(state):
        optimal_value = V_states[state][0]
        optimal_action = V_states[state][1]
        # print "opt action", optimal_action
        task = mdp.getTasksList()[optimal_action]
        next_state_tasks = list(state[0])[:]
        next_state_tasks[optimal_action] = 1
        next_state = (tuple(next_state_tasks), state[1] + task.getTimeCost())
        state = next_state
        optimal_tasks.append(task)

    optimal_policy = [task.getDescription() for task in optimal_tasks]
    time_elapsed = end - start
    
    return optimal_policy, iterations, time_elapsed

def get_Q_value(mdp, state, action, V_states):
    total = 0
    trans_states_and_probs = mdp.getTransitionStatesAndProbs(state, action)
    for pair in trans_states_and_probs:
        next_state = pair[0]
        tasks = next_state[0]
        time = next_state[1]
        prob = pair[1]
        next_state_value = V_states[next_state][0]
        total += prob * (mdp.getReward(state, action, next_state) + gamma * next_state_value)
    return total

def choose_action(mdp, state, V_states):
    possible_actions = mdp.getPossibleActions(state)   
    best_action = None
    best_value = -float('inf')
    if len(possible_actions) == 0:
        best_value = 0
    for a in possible_actions:
        q_value = get_Q_value(mdp, state, a, V_states)
        if q_value > best_value:
            best_value = q_value
            best_action = a
    return (best_value, best_action)

def policy_evaluation(mdp, policies):
    """
    given an MDP and a policy dictionary (from policy improvement)
    returns the V states for that policy for each state. V_states: {state: (V(s), action)}
    """
    

    
def policy_extraction(mdp, v_states):
    """
    given an MDP and V_states (from policy evaluation)
    returns the optimal policy (policy is dictionary{states: action index})
    """
    # for every state, pick the action corresponding to the highest Q-value
    policies = {}
    states = mdp.getStates()
    for state in states:
        best_action = choose_action(mdp, state, V_states)[0]
        policies[state] = best_action
    
    return policies

def policy_iteration(mdp):
    states = mdp.getStates()
    policy = {}
    new_policy = {}
    for state in states:
        new_policy[state] = 
    
    # repeat until policy converges
    while policy != new_policy:
        policy = new_policy
        v_states = policy_evaluation(mdp, policy)
        new_policy = policy_extraction(mdp, v_states)
        
    return policy