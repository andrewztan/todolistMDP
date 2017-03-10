from todolist import *
from mdp_solvers import *

goals1 = [
    Goal("Goal A", [
        Task("Task A1", 1)], 
        {1: 100},
        penalty=-1000),
    Goal("Goal B", [
        Task("Task B2", 1)], 
        {1: 10},
        penalty=-1000000)
]

goals2 = [
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
        penalty=-1)
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


# plotting number of tasks vs runtime with time kept constant at 500

end_time = 5
goals_list = [goals1, goals2, goals3, goals4, goals5, goals6]
iterations_list = []
times = []

<<<<<<< HEAD
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

todolist = ToDoList(goals3, start_time=0, end_time=end_time)
mdp = ToDoListMDP(todolist)

# run with value iteration
print 'value iteration'
vi_policy, vi_iterations, vi_time_elapsed = value_iteration(mdp)
print 'policy', vi_policy
print 'time (s)', vi_time_elapsed
print ''

# run with policy iteration
print 'policy iteration'
pi_policy, pi_iterations, pi_time_elapsed = policy_iteration(mdp)
print 'policy', pi_policy
print 'time (s)', pi_time_elapsed







