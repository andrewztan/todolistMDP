import time
import numpy as np 
from numpy import linalg as LA

def backward_induction(mdp):
    """
    Given a ToDoListMDP, perform value iteration/backward induction to find the optimal policy
    Input: MDP
    Output: Optimal policy, number of iterations, empirical runtime
    """
    V_states = {}
    linearized_states = mdp.getLinearizedStates()
    # print "linearized states:", linearized_states
    numTasks = len(linearized_states)
    for state in linearized_states:
        V_states[state] = (0, None)

    start = time.time()

    # Perform Backward Iteration (Value Iteration 1 Time)
    start = time.time()
    for state in linearized_states:
        V_states[state] = choose_action(mdp, state, V_states)        

    end = time.time()
    print 'time:', end - start

    start_state = mdp.getStartState()
    state = start_state
    optimal_tasks = []

    # Record Optimal Policy from start state
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
    
    return optimal_policy, 1, time_elapsed

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
