import time
import numpy as np 
from numpy import linalg as LA
from scipy.sparse import linalg as sLA
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from todolist import *
from mdp_solvers import *

def solve_big_goals(goals_list, end_time):
	total_time = 0

	descriptions = [goal.getDescription() for goal in goals_list]

	# transform into list of big goals only
	big_goals_list = [
		Goal(goal.getDescription(), 
		[Task(goal.getDescription(), goal.getDuration())], 
		goal.getRewardDict(), 
		penalty=goal.deadlinePenalty()) 
			for goal in goals_list]
			
	big_goals_policy, big_goals_time = solve_goals(big_goals_list, end_time)
	total_time += big_goals_time

	# for each goal, transform into list of tasks
	all_tasks_policy = []
	for desc in big_goals_policy:
		i = descriptions.index(desc)
		goal = [goals_list[i]]
		tasks_policy, tasks_time = solve_tasks(goal, end_time)
		all_tasks_policy += tasks_policy
		total_time += tasks_time

	return all_tasks_policy, total_time

def solve_tasks(goal, end_time):
	todolist = ToDoList(goal, start_time=0, end_time=end_time)
	mdp = ToDoListMDP(todolist)
	bi_policy, _, bi_time = backward_induction(mdp)
	return bi_policy, bi_time

def solve_goals(big_goals_list, end_time): 
	todolist = ToDoList(big_goals_list, start_time=0, end_time=end_time)
	big_mdp = ToDoListMDP(todolist)
	bi_policy, _, bi_time = backward_induction(big_mdp)
	return bi_policy, bi_time
