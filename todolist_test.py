from todolist import *

if __name__ == '__main__':

    goals = [
        Goal("Goal A", [
            Task("Task A1", 1), 
            Task("Task A2", 2)], 
            {1: 140, 3: 100, 7: 80},
            penalty=-10),
        Goal("Goal B", [
            Task("Task B1", 2),  
            Task("Task B3", 2)], 
            {1: 10, 3: 9, 4: 8},
            penalty=-1000),
    ]

    end_time = 20
    my_list = ToDoList(goals, start_time=0, end_time=end_time)
    mdp = ToDoListMDP(my_list)
    start_state = mdp.getStartState()

    action = mdp.getPossibleActions(start_state)[0]
    curr_state = mdp.getTransitionStatesAndProbs(start_state, action)[0][0]
    
    # action = mdp.getPossibleActions(next_state)[3]
    # next_state = mdp.getTransitionStatesAndProbs(next_state, action)[0][0]
    # print(next_state)

    print(mdp.getReward(start_state, action, curr_state))
    while not mdp.isTerminal(curr_state):
        print curr_state
        print "Possible Actions:", mdp.getPossibleActions(curr_state)
        action = mdp.getPossibleActions(curr_state)[0]
        transStatesAndProbs = mdp.getTransitionStatesAndProbs(curr_state, action)
        # print(transStatesAndProbs)
        next_state = transStatesAndProbs[0][0]
        print(mdp.getReward(curr_state, action, next_state))
        curr_state = next_state

    # create every single state possible
    numTasks = len(my_list.getTasks())
    states = {}
    for t in range(end_time + 1):
        bit_vectors = list(itertools.product([0, 1], repeat=numTasks))
        for bv in bit_vectors:
            # bv = list(bv)
            state = (bv, t)
            states[state] = (0, None)


    # perform value iteration with s iterations
    gamma = 1.0

    def sumTransitionStates(mdp, state, action, trans_states_and_probs, states):
        total = 0
        for pair in trans_states_and_probs:
            next_state = (tuple(pair[0][0]), pair[0][1])
            prob = pair[1]
            next_state_value = states[next_state][0]
            total += prob * (mdp.getReward(state, action, next_state) + gamma * next_state_value)
        return total
    print ""
    converged = False
    i = 0
    while not converged:
        print 'iteration', i
        i += 1
        next_V_states = {} 
        converged = True
        for state in states:
            possible_actions = mdp.getPossibleActions(state)   
            best_action = None
            best_value = -float('inf')
            for a in possible_actions:
                trans_states_and_probs = mdp.getTransitionStatesAndProbs(state, a)
                value = sumTransitionStates(mdp, state, a, trans_states_and_probs, states)
                if value > best_value:
                    best_value = value
                    best_action = a
            if len(possible_actions) == 0:
                best_value = 0
            next_V_states[state] = (best_value, best_action)

            old_state_value = states[state][0]
            new_state_value = best_value
            if abs(old_state_value - new_state_value) > 0.1:
                converged = False

        states = next_V_states

    # print(states)

    start_state = (tuple([0 for _ in range(numTasks)]), 0)
    state = start_state
    optimal_tasks = []
    while not mdp.isTerminal(state):
        optimal_value = states[state][0]
        optimal_action = states[state][1]
        print "opt action", optimal_action
        task = mdp.getTasksList()[optimal_action]
        next_state_tasks = list(state[0])[:]
        next_state_tasks[optimal_action] = 1
        next_state = (tuple(next_state_tasks), state[1] + task.getTimeCost())
        state = next_state
        optimal_tasks.append(task)

    print [task.getDescription() for task in optimal_tasks]











    # print("start state: " + str(start_state))
    # print(mdp.getPossibleActions(start_state))
    # print(mdp.getTransitionStatesAndProbs(start_state, 0))
    # my_list.printDebug()
    

