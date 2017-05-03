import random
import sys
import mdp
# import environment
# import util
# import optparse
import itertools
import math
import numpy as np
import time

class ToDoList():
    """
      To-do List
    """
    def __init__(self, goals, start_time=0, end_time=None, nongoal_val=1):
        self.goals = goals # list of goals
        self.completed_goals = set([goal for goal in self.goals if goal.isComplete()]) # set of completed goals
        self.incomplete_goals = set([goal for goal in self.goals if not goal.isComplete()])
        self.time = start_time # current time
        self.tasks = [] # list of tasks
        for goal in self.goals:
            self.tasks.extend(goal.getTasks())
        self.completed_tasks = set([task for task in self.tasks if task.isComplete()]) # set of completed tasks
        self.incomplete_tasks = set([task for task in self.tasks if not task.isComplete()])
        
        self.start_time = start_time
        if end_time == None:
            max_deadline = float('-inf')
            for goal in self.goals:
                goal_deadline = goal.getDeadline()
                if goal_deadline > max_deadline:
                    max_deadline = goal_deadline
            end_time = max_deadline + 1
        self.end_time = end_time

        # Nongoal task with time=1, prob=1
        self.nongoal_task = Task("Nongoal Task", time_cost=1, prob=1, reward=nongoal_val, nongoal=True)
        # create Nongoal "Goal" with Nongoal Task
        self.nongoal = Goal(description="Nongoal", tasks=[self.nongoal_task], reward={float('inf'): 0}, nongoal=True)


    # do a random action
    def action(self):
        i = int(random.random() * len(self.tasks))
        task = self.tasks[i]
        return self.action(task)

    # do a defined action, task
    def action(self, task):  
        reward = 0
        prev_time = self.time
        curr_time = self.time + task.getTimeCost()
        self.incrementTime(task.getTimeCost()) # increment the time so that we are working in the current time
        reward += self.doTask(task)
        reward += self.checkDeadlines(prev_time, curr_time)
        return reward

    def checkDeadlines(self, prev_time, curr_time):
        """
        Check which goals passed their deadline between prev_time and curr_time
        If goal passed deadline during task, incur penalty
        """
        penalty = 0
        # iterate through all incomplete goals
        for goal in self.incomplete_goals:
            # check 1) goal is now passed deadline at curr_time 2) goal was not passed deadline at prev_time 
            if curr_time > goal.getDeadline() and not prev_time > goal.getDeadline():
                penalty += goal.deadlinePenalty()
        return penalty
    
    def doTask(self, task):
        goal = task.getGoal()
        threshold = task.getProb()
        p = random.random()
        reward = 0
        reward += task.getReward()
        # check that task is completed on time and NOT a nongoal and goal was not already complete before task
        if p < threshold and self.time <= goal.getDeadline() and not task.isNongoal() and not self.isGoalComplete(goal):
            task.setCompleted(True)
            self.incomplete_tasks.discard(task)
            self.completed_tasks.add(task)
            # if completion of task completes the goal
            if self.isGoalComplete(goal):
                reward += goal.getReward(self.time) # goal completion reward
                self.incomplete_goals.discard(goal)
                self.completed_goals.add(goal)
        return reward

    def getNongoalVal(self):
        return self.nongoal_task.getReward()

    def getNongoalTask(self):
        return self.nongoal_task

    def isGoalComplete(self, goal):
        """
        Method for checking if goal is complete by checking if all tasks are complete
        """
        if goal.isComplete():
            return True
        for task in goal.tasks:
            if not task.isComplete():
                return False
        goal.setCompleted(True)
        return True

    def incrementTime(self, t=1):
        self.time += t

    def getTime(self):
        return self.time

    def getEndTime(self):
        return self.end_time

    def getTasks(self):
        return self.tasks

    def getGoals(self):
        return self.goals

    def addGoal(self, goal):
    	"""
    	Add an entire goal
    	"""
        self.goals.append(goal)
        self.tasks.extend(goal.getTasks())
        if goal.isComplete():
        	self.completed_goals.add(goal)
        else:
        	self.incomplete_goals.add(goal)


    def addTask(self, goal, task):
    	"""
    	Adds task to the specified goal
    	"""
        self.tasks.append(task)
        goal.addTask(task)

    # print ToDoList object
    def printDebug(self):
        print("Current Time: " + str(self.time))
        print("Goals: " + str(self.goals))
        print("Completed Goals: " + str(self.completed_goals))
        print("Tasks: " + str(self.tasks))
        print("Completed Tasks: " + str(self.completed_tasks))


class Goal():
    def __init__(self, description, tasks, reward, penalty=0, completed=False, nongoal=False):
        # parameters
        self.description = description # string description of goal
        self.tasks = tasks # list of Task objects
        for task in self.tasks:
            task.setGoal(self)
        self.reward = reward # dictionary that maps time (of completion) to reward value
        self.completed = completed # boolean for completion status
        self.deadline = max(reward.keys())
        self.penalty = penalty # penalty if goal is not completeed by the deadline
        self.nongoal = nongoal # boolean for whether goal is a nongoal

    ### Get functions ###
    def getTasks(self):
        return self.tasks

    # return reward based on time
    def getReward(self, time):
        if time > self.getDeadline():
            return 0
        times = sorted(self.reward.keys())
        t = next(val for x, val in enumerate(times) if val >= time)
        return self.reward[t]

    # return rewards dictionary that maps time of completion to reward value
    def getRewardDict(self):
        return self.reward

    # return deadline time, does not do any computation, just return self.deadline
    def getDeadline(self):
        return self.deadline

    def getDuration(self):
        time = 0
        for task in self.tasks:
            time += task.getTimeCost()
        return time

    def deadlinePenalty(self):
        return self.penalty

    def addTask(self, task):
        self.tasks.append(task)
        task.setGoal(self)

    def setCompleted(self, completed):
        self.completed = completed
        for task in self.tasks:
            task.setCompleted(True)

    # return completion status, does not do any computation, just return self.completed
    def isComplete(self):
        return self.completed

    def isNongoal(self):
        return self.nongoal

    def getDescription(self):
        return self.description


class Task():
    def __init__(self, description, time_cost, prob=1, reward=0, completed=False, goal=None, nongoal=False):
        # parameters
        self.description = description # string description of task
        self.goal = None # Goal object that encompasses this task
        self.prob = prob # probability of completion
        self.time_cost = time_cost # units of time required to perform a task
        self.reward = reward # integer value, usually 0
        self.completed = completed # boolean for completion status
        self.nongoal = nongoal # boolean for whether task is a nongoal task

    ### Get functions ###
    def getDescription(self):
        return self.description

    def getGoal(self):
        return self.goal

    # return the probability of completing the task
    def getProb(self):
        return self.prob

    # return the units of time required to perform a task
    def getTimeCost(self):
        return self.time_cost

    # return reward 
    def getReward(self):
        return self.reward

    # return completion status, does not do any computation, just return self.completed
    def isComplete(self):
        return self.completed

    def isNongoal(self):
        return self.nongoal

    ### Set functions ###
    def setGoal(self, goal):
        self.goal = goal

    def setCompleted(self, completed):
        self.completed = completed

    def copy(self):
        return Task(self.description, self.time_cost, self.prob, self.reward, self.completed, self.goal)


class ToDoListMDP(mdp.MarkovDecisionProcess):
    ###################
    ### MDP Methods ###
    ###################

    """
    state is represented as a tuple of:
    1. list of tasks (boolean for completion)
    2. time
    """

    def __init__(self, todolist, gamma=1.0):
        self.todolist = todolist
        self.start_state = self.getStartState()

        # nongoal
        self.nongoal_task = todolist.getNongoalTask()
        self.nongoal_val = self.nongoal_task.getReward()

        # create mapping of indices to tasks - represented as list
        self.index_to_task = list(todolist.getTasks())
        self.index_to_task.append(self.nongoal_task) # add nongoal task
        # create mapping of tasks to indices - represented as dict
        self.task_to_index = {}
        for i in range(len(self.index_to_task)):
            task = self.index_to_task[i]
            self.task_to_index[task] = i

        # creating Goals and their corresponding tasks pointers
        self.goals = self.todolist.getGoals()
        self.goal_to_indices = {}
        for goal in self.goals:
            task_indices = [self.task_to_index[task] for task in goal.getTasks()]
            self.goal_to_indices[goal] = task_indices

        # parameters
        self.livingReward = 0.0
        self.noise = 0.0

        self.gamma = gamma

        self.states = []
        numTasks = len(todolist.getTasks())
        for t in range(self.todolist.getEndTime() + 2):
            bit_vectors = itertools.product([0, 1], repeat=numTasks)
            for bv in bit_vectors:
                state = (bv, t)
                self.states.append(state)

        self.state_to_index = {self.states[i]: i for i in range(len(self.states))}

        self.reverse_DAG = MDPGraph(self)
        self.linearized_states = self.reverse_DAG.linearize()

        self.V_states, self.optimal_policy = self.value_and_policy_functions()

        # Pseudo-rewards
        self.pseudorewards = {} # keys are (s, a, s'). values are PR(s, a, s')
        self.transformed_pseudorewards = {} # keys are (s, a, s'). values are PR'(s, a, s')
        self.calculatePseudorewards() # calculate pseudorewards for each state
        # self.transformPseudorewards() # applies linear transform PR to PR'

    def getPseudorewards(self):
        """ getter method for pseudorewards
        pseudorewards is stored as a dictionary, 
        where keys are tuples (s, s') and values are PR'(s, a, s')
        """
        return self.pseudorewards

    def getTransformedPseudorewards(self):
        return self.transformed_pseudorewards

    def getExpectedPseudorewards(self, state, action):
        """
        Return the expected Pseudorewards of a (state, action) pair
        """
        expected_pr = 0
        trans_states_and_probs = self.getTransitionStatesAndProbs(state, action)
        for pair in trans_states_and_probs:
            next_state, prob = pair
            expected_pr += prob * self.pseudorewards[(state, action, next_state)]
        return expected_pr

    def getExpectedTransformedPseudorewards(self, state, action):
        """
        Return the expected Pseudorewards of a (state, action) pair
        """
        expected_pr = 0
        trans_states_and_probs = self.getTransitionStatesAndProbs(state, action)
        for pair in trans_states_and_probs:
            next_state, prob = pair
            expected_pr += prob * self.transformed_pseudorewards[(state, action, next_state)]
        return expected_pr

    def calculatePseudorewards(self):
        """
        private method for calculating untransformed pseudorewards PR
        """
        for state in self.states:
            for action in self.getPossibleActions(state):
                for next_state, prob in self.getTransitionStatesAndProbs(state, action):
                    r = self.getReward(state, action, next_state)
                    pr = self.V_states[next_state][0] - self.V_states[state][0] + r
                    self.pseudorewards[(state, action, next_state)] = pr

    def transformPseudorewards(self):
        """
        linearly transforms PR to PR' such that:
            - PR' > 0 for all optimal actions
            - PR' <= 0 for all suboptimal actions
        """
        highest = -float('inf')
        sec_highest = -float('inf')

        for trans in self.pseudorewards:
            pr = self.pseudorewards[trans]
            if pr > highest:
                sec_highest = highest
                highest = pr
            elif pr > sec_highest and pr < highest:
                sec_highest = pr
        
        # for state in self.states:
        #     for action in self.getPossibleActions(state):
        #         for 


        print highest
        print sec_highest

        alpha = (highest + sec_highest) / 2
        print alpha
        beta = 1
        if alpha <= 1.0: beta = 10

        for trans in self.pseudorewards:
            pr = self.pseudorewards[trans]
            self.transformed_pseudorewards[trans] = (alpha + pr) * beta
        
    def getLinearizedStates(self):
        return self.linearized_states

    def getGamma(self):
        return self.gamma

    def getTasksList(self):
        # print len(self.index_to_task)
        return self.index_to_task

    def setLivingReward(self, reward):
        """
        The (negative) reward for exiting "normal" states.
        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.livingReward = reward

    def setNoise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise

    def tasksToBinary(self, tasks):
        """
        Convert a list of Task objects to a bit vector with 1 being complete and 0 if not complete. 
        """
        binary_tasks = tuple([1 if task.isComplete() else 0 for task in tasks])
        return binary_tasks

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        start_state = self.tasksToBinary(self.todolist.getTasks())
        return (start_state, 0)

    def getStates(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return self.states

    def getStateIndex(self, state):
        return self.state_to_index[state]

    def getPossibleActions(self, state):
        """
        Return list of possible actions from 'state'.
        Returns a list of indices
        """
        tasks = state[0]
        currentTime = state[1]
        if not self.isTerminal(state): 
            # possible_actions = [i for i, task in enumerate(tasks) if (task == 0 and self.isTaskActive(self.index_to_task[i], currentTime))]
            # possible_actions = [i for i, task in enumerate(tasks) if (task == 0 and self.isTaskActive(self.index_to_task[i], currentTime + self.index_to_task[i].getTimeCost()))]
            possible_actions = [i for i, task in enumerate(tasks) if task == 0]
            possible_actions.append(-1) # append non-goal action, represented by -1
        else:
            possible_actions = []
        return possible_actions

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        next_states_probs = []

        # action is nongoal action
        if action == -1:
            binary_tasks = list(state[0])[:]
            new_time = state[1] + 1
            next_states_probs.append(((tuple(binary_tasks), new_time), 1))
            return next_states_probs

        # action is the index that is passed in 
        task = self.index_to_task[action]
        binary_tasks = list(state[0])[:] # make a modifiable copy of tasks
        new_time = state[1] + task.getTimeCost()

        if new_time > self.todolist.getEndTime():
            new_time = self.todolist.getEndTime() + 1

        # state for not completing task
        tasks_no_completion = binary_tasks[:]
        if 1 - task.getProb() > 0:
            next_states_probs.append(((tuple(tasks_no_completion), new_time), 1 - task.getProb()))
        # state for completing task
        tasks_completion = binary_tasks[:]
        tasks_completion[action] = 1
        if task.getProb() > 0:
            next_states_probs.append(((tuple(tasks_completion), new_time), task.getProb()))

        return next_states_probs

    def getReward(self, state, action, nextState):
        """
        Get the reward for the state, action, nextState transition.
        state: (list of tasks, time)
        action: integer of index of tax
        nextState: (list of tasks, time)

        Not available in reinforcement learning.
        """
        reward = 0
        task = self.index_to_task[action]
        goal = task.getGoal()
        prev_tasks = state[0]
        prev_time = state[1]
        next_tasks = nextState[0]
        next_time = nextState[1]

        # reward from doing a task
        reward += task.getReward() 

        # action is NOT nongoal task

        if action != -1:
            # reward for goal completion
            if next_tasks[action] == 1:
                if self.isGoalCompleted(goal, nextState) and self.isGoalActive(goal, next_time):
                    reward += goal.getReward(next_time)

        # penalty for missing a deadline
        for goal in self.goals:
            if not self.isGoalCompleted(goal, state) and self.isGoalActive(goal, prev_time) and not self.isGoalActive(goal, next_time):
                # if a deadline passed during time of action, add reward (penalty)
                reward += goal.deadlinePenalty()
        
        return reward

    def isTerminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        tasks = state[0]
        time = state[1]
        # check if the global end time is reached or if all tasks are completed
        if time > self.todolist.getEndTime() or not 0 in tasks:
            return True
        # check if there are any goals that are still active and not completed
        for goal in self.goals:
            if self.isGoalActive(goal, time) and not self.isGoalCompleted(goal, state):
                return False
        return True

    def isGoalActive(self, goal, time):
        """
        Given a Goal object and a time
        Check if the goal is still active at that time
        Note: completed goal is still considered active if time has not passed the deadline 
        """
        active = time <= goal.getDeadline() and time <= self.todolist.getEndTime()
        return active

    def isGoalCompleted(self, goal, state):
        """
        Given a Goal object and current state
        Check if the goal is completed 
        """
        tasks = state[0]
        goal_indices = self.goal_to_indices[goal]
        goal_tasks = [tasks[i] for i in goal_indices]
        return not 0 in goal_tasks 

    def isTaskActive(self, task, time):
        """
        Given a Task object and a time
        Check if the goal is still active at that time 
        """
        goal = task.getGoal()
        return self.isGoalActive(goal, time)

    def backward_induction(self, time=False):
        """
        Given a ToDoListMDP, perform value iteration/backward induction to find the optimal policy
        Input: ToDoListMDP
        Output: Optimal policy (and time elapsed if specified)
        """
        start = time.time()

        V_states = {} # maps state to (value, action)
        linearized_states = self.getLinearizedStates()
        # print "linearized states:", linearized_states
        numTasks = len(linearized_states)
        for state in linearized_states:
            V_states[state] = (0, None)

        # Perform Backward Iteration (Value Iteration 1 Time)
        start = time.time()

        for state in linearized_states:
            V_states[state] = choose_action(mdp, state, V_states)        

        end = time.time()

        optimal_policy = {}
        for state in V_states:
            optimal_policy[state] = V_states[state][1]

        time_elapsed = end - start

        # mdp.calculatePseudorewards(V_states)
        
        if time:
            return optimal_policy, time_elapsed
        else:
            return optimal_policy

    def get_Q_value(self, state, action, V_states):
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
        trans_states_and_probs = self.getTransitionStatesAndProbs(state, action)
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
            Q_value += prob * (self.getReward(state, action, next_state) + self.getGamma() * next_state_value)
        return Q_value

    def getValueAndAction(self, state, V_states):
        """
        Input: 
        mdp: ToDoList MDP
        state: current state (tasks, time)
        V_states: dictionary mapping states to current best (value, action)
        Output:
        best_value: value of state state
        best_action: index of action that yields highest value at current state
        """
        possible_actions = self.getPossibleActions(state)   
        best_action = None
        best_value = -float('inf')
        if self.isTerminal(state):
            best_value = 0
            best_action = 0
        for a in possible_actions:
            q_value = self.get_Q_value(state, a, V_states)
            if q_value > best_value:
                best_value = q_value
                best_action = a
        return (best_value, best_action)


    def value_and_policy_functions(self):
        """
        Given a ToDoListMDP, perform value iteration/backward induction to find the optimal value function
        Input: ToDoListMDP
        Output: Dictionary of optimal value of each state
        """
        V_states = {} # maps state to (value, action)
        lin_states = self.linearized_states
        # print "linearized states:", linearized_states
        numTasks = len(lin_states)
        for state in lin_states:
            V_states[state] = (0, None)

        # Perform Backward Iteration (Value Iteration 1 Time)
        for state in lin_states:
            V_states[state] = self.getValueAndAction(state, V_states)        

        optimal_policy = {}
        for state in V_states:
            optimal_policy[state] = V_states[state][1]

        return V_states, optimal_policy

    def get_optimal_policy_dict(self):
        """
        Returns the mapping of state to the optimal policy at that state
        """
        return self.optimal_policy

    def get_optimal_policy(self, state):
        return self.optimal_policy[state]

    def get_value_function(self):
        return self.V_states

    def get_state_value(self, state):
        return self.V_states[state][0]


class MDPGraph():
    def __init__(self, mdp):
        start = time.time()
        # print 'building reverse graph'
        self.mdp = mdp
        self.vertices = []
        self.edges = {}
        self.preorder = {}
        self.postorder = {}
        for state in mdp.getStates():
            self.vertices.append(state)
            if state not in self.edges:
                self.edges[state] = set()
            for action in mdp.getPossibleActions(state):
                for pair in mdp.getTransitionStatesAndProbs(state, action):
                    next_state, prob = pair
                    if next_state not in self.edges:
                        self.edges[next_state] = set()
                    self.edges[next_state].add(state)
        # print 'done building reverse graph'
        end = time.time()
        # print 'time:', end - start
        # print '' 
        # print 'vertices', self.vertices
        # print 'edges', self.edges


    def getVertices(self):
        return self.vertices

    def getEdges(self):
        return self.edges

    def linearize(self):
        """
        Returns list of states in topological order
        """

        self.dfs() # run dfs to get postorder
        # postorder_dict = self.dfs(self.reverse_graph)
        postorder = [(v, -self.postorder[v]) for v in self.postorder]
        dtype = [('state', tuple), ('postorder', int)]
        a = np.array(postorder, dtype=dtype)
        reverse_postorder = np.sort(a, order='postorder')

        self.linearized_states = [state for (state, i) in reverse_postorder]
        return self.linearized_states

    def dfs(self):
        visited = {}
        self.counter = 1

        def explore(v):
            visited[v] = True
            self.preorder[v] = self.counter
            self.counter += 1
            for u in self.edges[v]:
                if not visited[u]:
                    explore(u)
            self.postorder[v] = self.counter
            self.counter += 1

        for v in self.vertices:
            visited[v] = False

        for v in self.vertices:
            if not visited[v]:
                explore(v)

