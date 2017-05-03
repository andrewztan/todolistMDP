from todolist import *
from mdp_solvers import *
import random
# from mdp_pol_iter import *
# from mdp_val_iter import *
# from mdp_backward_induction import *
# from mdp_goals_solver import *

goals1 = [
    Goal("Goal A", [
        Task("Task A1", 1)], 
        {1: 100},
        penalty=-1000),
    Goal("Goal B", [
        Task("Task B2", 1)], 
        {1: 10},
        penalty=-1000000), 
]

goals2 = [
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 1),  
        Task("Task B2", 1)], 
        {1: 10, 10: 10},
        penalty=0)
]

goals3 = [
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),  
        Task("Task B2", 2)], 
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),  
        Task("Task C2", 3)], 
        {1: 10, 6: 100},
        penalty=-1), 
    Goal("Goal D", [
        Task("Task D1", 3),  
        Task("Task D2", 3)], 
        {20: 100, 40: 10},
        penalty=-10),
]

goals4 = [
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),  
        Task("Task B2", 2)], 
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),  
        Task("Task C2", 3)], 
        {1: 10, 6: 100},
        penalty=-1), 
    Goal("Goal D", [
        Task("Task D1", 3),  
        Task("Task D2", 3)], 
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),  
        Task("Task E2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
]

goals5 = [
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),  
        Task("Task B2", 2)], 
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),  
        Task("Task C2", 3)], 
        {1: 10, 6: 100},
        penalty=-1), 
    Goal("Goal D", [
        Task("Task D1", 3),  
        Task("Task D2", 3)], 
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),  
        Task("Task E2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),  
        Task("Task F2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
]

goals6 = [
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal B", [
        Task("Task B1", 2),  
        Task("Task B2", 2)], 
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal C", [
        Task("Task C1", 3),  
        Task("Task C2", 3)], 
        {1: 10, 6: 100},
        penalty=-1), 
    Goal("Goal D", [
        Task("Task D1", 3),  
        Task("Task D2", 3)], 
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),  
        Task("Task E2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),  
        Task("Task F2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal G", [
        Task("Task G1", 3),  
        Task("Task G2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
]

goals6b = [
    Goal("Goal B", [
        Task("Task B1", 2),  
        Task("Task B2", 2)], 
        {1: 10, 10: 10},
        penalty=0),
    Goal("Goal A", [
        Task("Task A1", 1), 
        Task("Task A2", 1)], 
        {10: 100},
        penalty=-10),
    Goal("Goal C", [
        Task("Task C1", 3),  
        Task("Task C2", 3)], 
        {1: 10, 6: 100},
        penalty=-1), 
    Goal("Goal D", [
        Task("Task D1", 3),  
        Task("Task D2", 3)], 
        {20: 100, 40: 10},
        penalty=-10),
    Goal("Goal E", [
        Task("Task E1", 3),  
        Task("Task E2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal F", [
        Task("Task F1", 3),  
        Task("Task F2", 3)], 
        {60: 100, 70: 10},
        penalty=-110),
    Goal("Goal G", [
        Task("Task G1", 3),  
        Task("Task G2", 3)], 
        {60: 100, 70: 10},
        penalty=-110)
]


goals7 = [
    Goal("CS HW", [
        Task("CS 1", time_cost=1, prob=0.9), 
        Task("CS 2", time_cost=2, prob=0.8)], 
        {7: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_cost=4, prob=0.95),  
        Task("EE 2", time_cost=2, prob=0.9)], 
        {14: 100},
        penalty=-200)
    # Goal("Goal C", [
    #     Task("Task C1", 3),  
    #     Task("Task C2", 3)], 
    #     {1: 10, 6: 100},
    #     penalty=-1)
]


simple_goals_deterministic = [
    Goal("CS HW", [
        Task("CS 1", time_cost=1, prob=1), 
        Task("CS 2", time_cost=1, prob=1)], 
        {3: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_cost=1, prob=1),  
        Task("EE 2", time_cost=2, prob=1)], 
        {4: 100},
        penalty=-200)
]

simple_goals_probabilistic = [
    Goal("CS HW", [
        Task("CS 1", time_cost=1, prob=0.9), 
        Task("CS 2", time_cost=1, prob=0.8)], 
        {4: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_cost=1, prob=0.95),  
        Task("EE 2", time_cost=2, prob=0.95)], 
        {5: 100},
        penalty=-200)
]

simple_goals_probabilistic2 = [
    Goal("CS HW", [
    	Task("CS 1", time_cost=1, prob=0.9), 
    	Task("CS 2", time_cost=1, prob=0.8)], 
        {3: 100},
        penalty=-200)
]


# plotting number of tasks vs runtime with time kept constant at 500


goals_list = [goals1, goals2, goals3, goals4, goals5, goals6]
iterations_list = []
times = []


todolist = ToDoList(simple_goals_probabilistic2, start_time=0, nongoal_val=1)
mdp = ToDoListMDP(todolist)

# run with value iteration
# print 'value iteration'
# vi_policy, vi_iterations, vi_time_elapsed = value_iteration(mdp)
# print 'policy', vi_policy
# print 'time (s)', vi_time_elapsed
# print ''


# get optimal policy with backward induction
print 'backward induction'
bi_policy = backward_induction(mdp, printTime=False)

print "Optimal Policy:"
print mdp.get_optimal_policy_dict()
print "Value Function:"
print mdp.get_value_function()
print "Pseudo-rewards:"
print sorted(mdp.getPseudorewards().values())[::-1]
print mdp.getPseudorewards()

print ""
print ""
print mdp.getReward(((1, 0), 2), 1, ((1, 1), 3))
print mdp.get_state_value(((1, 0), 2))
print mdp.get_state_value(((1, 1), 3))
# print 'policy', bi_policy
# print 'time (s)', bi_time_elapsed


"""
start_state = mdp.getStartState()
print bi_policy[start_state]
for action in mdp.getPossibleActions(start_state):
	print action, mdp.getPseudorewards(start_state, action)

state = ((0, 0, 0, 0), 1)
print bi_policy[state]
for action in mdp.getPossibleActions(state):
	print action, mdp.getPseudorewards(state, action)

state = ((0, 0, 0, 0), 2)
print bi_policy[state]
for action in mdp.getPossibleActions(state):
	print action, mdp.getPseudorewards(state, action)
"""


"""
# test our optimal policy
trials = 1
for i in range(trials):
    print 'Trial:', i
    start_state = mdp.getStartState()
    state = start_state
    performed_tasks = []
    reward = 0
    # Record policy from start state
    while not mdp.isTerminal(state):
        print state
        optimal_action = bi_policy[state]
        # print "opt action", optimal_action
        task = mdp.getTasksList()[optimal_action]
        print task.getDescription()
        performed_tasks.append((state, task))
        p = random.random()
        next_state_tasks = list(state[0])
        if p < task.getProb() and optimal_action != -1:
            next_state_tasks[optimal_action] = 1
        next_state = (tuple(next_state_tasks), state[1] + task.getTimeCost())
        reward += mdp.getReward(state, optimal_action, next_state)
        print 'Reward:', reward
        state = next_state
 """

    # for state, task in performed_tasks:
    #     print state
    #     print task.getDescription()
    
    # optimal_policy = [task.getDescription() for state, task in performed_tasks]
    # print optimal_policy



"""
# run with goal mdp and then task mdp
policy, time = solve_big_goals(goals6, end_time)
print 'policy', policy
print 'time', time
"""

# # run with policy iteration
# print 'policy iteration'
# pi_policy, pi_iterations, pi_time_elapsed = policy_iteration(mdp)
# print 'policy', pi_policy
# print 'time (s)', pi_time_elapsed






