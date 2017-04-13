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
    def __init__(self, goals, start_time=0, end_time=100, nongoal_val=1):
        # layout
        self.goals = goals # list of goals
        # self.completed_goals = [goal for goal in self.goals if goal.isComplete()] # list of completed goals
        # self.incompleted_goals = [goal for goal in self.goals if not goal.isComplete()]
        # incomplete goals attribute goes here ***
        self.time = start_time # current time
        self.tasks = [] # list of tasks
        for goal in self.goals:
            self.tasks.extend(goal.getTasks())
        # self.completed_tasks = [task for task in self.tasks if task.isComplete()] # list of completed tasks
        # self.incompleted_tasks = [task for task in self.tasks if not task.isComplete()]
        # incompleted tasks attribute goes here ***
        
        self.start_time = start_time
        self.end_time = end_time

        # create "Nongoal" goal with tasks and add to goals
        # self.nongoal = Goal("Nongoal", [], {end_time: 0}, 0, True, False)
        self.nongoal_val = nongoal_val

        # Nongoal task
        self.nongoal_task = Task("Nongoal Task", 1, 1, nongoal_val)
        # for i in range(1, end_time + 1):
        #     self.nongoal.addTask(Task("Non-goal, Time: " + str(i), i, 1, i * nongoal_val, False, self.nongoal))
        # self.goals.append(nongoal)


    # do a random action
    def action(self):
        i = int(random.random() * len(self.tasks))
        task = self.tasks[i]
        return self.action(task)

    # do a defined action, task
    def action(self, task):  
        reward = 0
        self.incrementTime(task.getCost()) # first, increment the time so that we are working in the current time
        reward += self.doTask(task)
        reward += self.checkDeadlines()
        return reward

    def checkDeadlines(self):
        penalty = 0
        for goal in self.goals:
            if self.time > goal.getDeadline():
                penalty += goal.deadlinePenalty()
                goal.setCompleted(True)
                # something about incomplete goals and tasks ***
        return penalty
    
    def doTask(self, task):
        p = task.getProb()
        reward = 0
        reward += task.getReward()
        if random.random() < p and self.time <= task.getGoal().getDeadline():
            # task completed
            task.setCompleted(True)
            # self.completed_tasks.append(task), add this back later when needed ***

            # if non-goal task, add back a task with same attributes. non-goal will never be complete
            if task.getGoal().isNongoal():
                copy_task = task.copy()
                task.getGoal().addTask(copy_task)
            # if goal complete, get reward
            elif self.isGoalComplete(task.getGoal()):
                reward += self.getGoalReward(task.getGoal())
                # soemthing about completed goals ***
        return reward

    def getNongoalVal(self):
        return self.nongoal_val

    def getNongoalTask(self):
        return self.nongoal_task

    def getGoalReward(self, goal):
        return goal.getReward(self.time)

    def isGoalComplete(self, goal):
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
        self.goals.append(goal)
        self.tasks.extend(goal.getTasks())

    def addTask(self, goal, task):
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
    def __init__(self, description, tasks, reward, penalty=0, nongoal=False, completed=False):
        # parameters
        self.description = description # string description of goal
        self.tasks = tasks # list of Task objects
        for task in self.tasks:
            task.setGoal(self)
        self.reward = reward # dictionary that maps time (of completion) to reward value
        self.completed = completed # boolean for completion status
        self.deadline = max(reward.keys())
        self.penalty = penalty # penalty if goal is not completeed by the deadline
        self.nongoal = nongoal

    ### Get functions ###
    def getTasks(self):
        return self.tasks

    # return reward based on time
    def getReward(self, time):
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
    def __init__(self, description, time_cost, prob=1, reward=0, completed=False, goal=None):
        # parameters
        self.description = description # string description of task
        self.goal = None # Goal object that encompasses this task
        self.prob = prob # probability of completion
        self.time_cost = time_cost # units of time required to perform a task
        self.reward = reward # integer value, usually 0
        self.completed = completed # boolean for completion status

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

    def getCost(self):
        return self.time_cost

    # return completion status, does not do any computation, just return self.completed
    def isComplete(self):
        return self.completed

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

        self.nongoal_val = todolist.getNongoalVal()
        self.nongoal_task = todolist.getNongoalTask()
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

        # create a list of deadline mappings (Do we need this if goal object has deadlines already?)
        self.deadlineMaps = {goal: goal.getRewardDict() for goal in self.goals}

        # create an initially empty set of inactive goals
        self.inactiveGoals = set()

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

    def getPseudorewards(self):
        """ getter method for pseudorewards
        pseudorewards is stored as a dictionary, 
        where keys are tuples (s, s') and values are PR'(s, a, s')
        """
        return self.pseudorewards

    def calculatePseudorewards(self, v_states):
        """
        private method for calculating untransformed pseudorewards PR
        """
        self.pseudorewards = {} # keys are (s, s'). values are PR(s, a, s')
        for state in self.states:
            actions = self.getPossibleActions(state)
            for a in actions:
                trans_states_and_probs = self.getTransitionStatesAndProbs(state, a)
                for pair in trans_states_and_probs:
                    next_state, prob = pair
                    r = self.getReward(state, a, next_state)
                    pr = v_states[next_state][0] - v_states[state][0] + r
                    self.pseudorewards[(state, next_state)] = pr

        # applies linear transform PR to PR'
        self.transformPseudorewards()

    def transformPseudorewards(self):
        """
        linearly transforms PR to PR' such that:
            - PR' > 0 for all optimal actions
            - PR' <= 0 for all suboptimal actions
        """
        highest = -float('inf')
        sec_highest = -float('inf')

        # a = list(set(self.pseudorewards.values()))
        # list.sort(a)
        # print a

        for trans in self.pseudorewards:
            pr = self.pseudorewards[trans]
            if pr > highest:
                sec_highest = highest
                highest = pr
            elif pr > sec_highest and pr < highest:
                sec_highest = pr

        # print highest
        # print sec_highest

        alpha = max(range(abs(highest), int(math.floor(abs(sec_highest)))))
        beta = 1
        if alpha <= 1.0: beta = 10

        for trans in self.pseudorewards:
            pr = self.pseudorewards[trans]
            self.pseudorewards[trans] = (alpha + pr) * beta
        
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
        if currentTime <= self.todolist.getEndTime(): 
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

