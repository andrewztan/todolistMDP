from todolist import *

if __name__ == '__main__':

    goals = [
        Goal("Goal A", [
            # Task("Task A1", 1), 
            Task("Task A1", 1)], 
            {1: 100},
            penalty=-10),
        Goal("Goal B", [
            # Task("Task B1", 1),  
            Task("Task B2", 1)], 
            {1: 10},
            penalty=-1000)
    ]

    goals2 = [
        Goal("Goal A", [
            Task("Task A1", 1), 
            Task("Task A2", 1)], 
            {20: 100},
            penalty=-10),
        Goal("Goal B", [
            Task("Task B1", 2),  
            Task("Task B2", 2)], 
            {1: 10, 10: 1},
            penalty=-1000),
        Goal("Goal C", [
            Task("Task C1", 3),  
            Task("Task C2", 3)], 
            {1: 10, 6: 1},
            penalty=-10000)
    ]

    end_time = 20
    my_list = ToDoList(goals2, start_time=0, end_time=end_time)
    mdp = ToDoListMDP(my_list)
    start_state = mdp.getStartState()

    action = mdp.getPossibleActions(start_state)[0]
    curr_state = mdp.getTransitionStatesAndProbs(start_state, action)[0][0]

    # action = mdp.getPossibleActions(next_state)[3]
    # next_state = mdp.getTransitionStatesAndProbs(next_state, action)[0][0]
    # print(next_state)

    # print(mdp.getReward(start_state, action, curr_state))
    # while not mdp.isTerminal(curr_state):
    #     print curr_state
    #     print "Possible Actions:", mdp.getPossibleActions(curr_state)
    #     action = mdp.getPossibleActions(curr_state)[0]
    #     transStatesAndProbs = mdp.getTransitionStatesAndProbs(curr_state, action)
    #     # print(transStatesAndProbs)
    #     next_state = transStatesAndProbs[0][0]
    #     print(mdp.getReward(curr_state, action, next_state))
    #     curr_state = next_state

    # create every single state possible

    # print mdp.getPossibleActions(((1, 0), 1))
    # print mdp.getTransitionStatesAndProbs(((1, 0), 1), 1)
    # print mdp.getReward(((1, 0), 1), 1, ((1, 1), 2))

    numTasks = len(my_list.getTasks())
    V_states = {}
    for t in range(end_time + 2):
        bit_vectors = list(itertools.product([0, 1], repeat=numTasks))
        # print bit_vectors
        for bv in bit_vectors:
            # bv = list(bv)
            state = (bv, t)
            V_states[state] = (0, None)

    # perform value iteration with s iterations
    gamma = 1.0

    def sumTransitionStates(mdp, state, action, V_states):
        total = 0
        trans_states_and_probs = mdp.getTransitionStatesAndProbs(state, action)
        for pair in trans_states_and_probs:
            next_state = pair[0]
            tasks = next_state[0]
            time = next_state[1]
            prob = pair[1]
            # print pair
            # print tasks
            # next_state = (tuple(tasks), time)
            # print next_state
            next_state_value = V_states[next_state][0]
            total += prob * (mdp.getReward(state, action, next_state) + gamma * next_state_value)
        # print total
        return total

    converged = False
    i = 0
    print V_states
    while not converged:
        print 'iteration', i
        i += 1
        next_V_states = {} 
        converged = True
        for state in V_states:
            # print state
            possible_actions = mdp.getPossibleActions(state)   
            best_action = None
            best_value = -float('inf')
            if len(possible_actions) == 0:
                best_value = 0
                # continue
            for a in possible_actions:
                value = sumTransitionStates(mdp, state, a, V_states)
                # print value
                if value > best_value:
                    best_value = value
                    best_action = a
            # print "state", state
            # print "value", best_value
            # print "policy", best_action
            next_V_states[state] = (best_value, best_action)

            old_state_value = V_states[state][0]
            new_state_value = best_value
            if abs(old_state_value - new_state_value) > 0.1:
                converged = False
        V_states = next_V_states
        # print V_states

    # print(V_states)

    start_state = (tuple([0 for _ in range(numTasks)]), 0)
    state = start_state
    optimal_tasks = []
    while not mdp.isTerminal(state):
        optimal_value = V_states[state][0]
        optimal_action = V_states[state][1]
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
    

