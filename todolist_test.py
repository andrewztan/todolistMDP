from todolist import *
from mdp_pol_iter import *
from mdp_val_iter import *
from mdp_backward_induction import *
from mdp_goals_solver import *

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
        Task("CS 1", time_cost=2, prob=0.9), 
        Task("CS 2", time_cost=3, prob=0.8)], 
        {7: 5},
        penalty=-10),
    Goal("EE Project", [
        Task("EE 1", time_cost=7, prob=0.5),  
        Task("EE 2", time_cost=2, prob=1)], 
        {14: 10},
        penalty=-20)
    # Goal("Goal C", [
    #     Task("Task C1", 3),  
    #     Task("Task C2", 3)], 
    #     {1: 10, 6: 100},
    #     penalty=-1)
]


# plotting number of tasks vs runtime with time kept constant at 500


goals_list = [goals1, goals2, goals3, goals4, goals5, goals6]
iterations_list = []
times = []

"""
for i in range(3):
=======
for i in range(1):
>>>>>>> 94cc6dcb1ab2c3bea2a18b5db1058347294f71da
    print('goals', i+1)
    goals = goals_list[i]
    todolist = ToDoList(goals, start_time=0, end_time=end_time)
    mdp = ToDoListMDP(todolist)
    policy, iterations, time_elapsed = value_iteration(mdp)
    iterations_list.append(iterations)
    times.append(time_elapsed)
    print(time_elapsed)
    print(policy)
    print()
"""

todolist = ToDoList(goals7, start_time=0, end_time=20, nongoal_val=1)
print 'mdp'
mdp = ToDoListMDP(todolist)
print 'todo.getTasks'
print mdp.getStartState() 

# run with value iteration
# print 'value iteration'
# vi_policy, vi_iterations, vi_time_elapsed = value_iteration(mdp)
# print 'policy', vi_policy
# print 'time (s)', vi_time_elapsed
# print ''


# run with backward induction
print 'backward induction'
bi_policy, bi_iterations, bi_time_elapsed = backward_induction(mdp)
print 'policy', bi_policy
print 'time (s)', bi_time_elapsed
print ''

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





