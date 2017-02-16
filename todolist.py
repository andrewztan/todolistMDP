import random
import sys
import mdp
# import environment
import util
# import optparse

class ToDoList():
    """
      To-do List
    """
    def __init__(self, goals, start_time=0, end_time=100, non_goal_c=1):
        # layout
        self.goals = goals # list of goals
        self.completed_goals = [goal for goal in self.goals if goal.isComplete()] # list of completed goals
        self.incompleted_goals = [goal for goal in self.goals if not goal.isComplete()]
        # incomplete goals attribute goes here ***
        self.time = start_time # current time
        self.tasks = [] # list of tasks
        for goal in self.goals:
            self.tasks.extend(goal.getTasks())
        self.completed_tasks = [task for task in self.tasks if task.isComplete()] # list of completed tasks
        self.incompleted_tasks = [task for task in self.tasks if not task.isComplete()]
        # incompleted tasks attribute goes here ***

        

        self.start_time = start_time
        self.end_time = end_time

        # create "Non Goal" goal with tasks and add to goals
        self.non_goal = Goal("Non-goal", [], {end_time: 0}, 0, True, False)
        self.non_goal_c = non_goal_c
        for i in range(1, end_time + 1):
            self.non_goal.addTask(Task("Non-goal, Time: " + str(i), i, 1, i * non_goal_c, False, self.non_goal))
        # self.goals.append(non_goal)


    # do a random action
    def action(self):
        i = int(random.random() * len(self.tasks))
        task = self.tasks[i]
        return self.action(task)

    # do a defined action, task
    def action(self, task):  
        reward = 0
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
        self.incrementTime(task.getCost()) # first, increment the time so that we are working in the current time
        p = task.getProb()
        reward = 0
        reward += task.getReward()
        if random.random() < p:
            # task completed
            task.setCompleted(True)
            # self.completed_tasks.append(task), add this back later when needed ***

            # if non-goal task, add back a task with same attributes. non-goal will never be complete
            if task.getGoal().isNonGoal():
                copy_task = task.copy()
                task.getGoal().addTask(copy_task)
            # if goal complete, get reward
            elif self.isGoalComplete(task.getGoal()):
                reward += self.getGoalReward(task.getGoal())
                # soemthing about completed goals ***
        return reward

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
    def __init__(self, description, tasks, reward, penalty=0, non_goal=False, completed=False):
        # parameters
        self.description = description # string description of goal
        self.tasks = tasks # list of Task objects
        for task in self.tasks:
            task.setGoal(self)
        self.reward = reward # dictionary that maps time (of completion) to reward value
        self.completed = completed # boolean for completion status
        self.deadline = max(reward.keys())
        self.penalty = penalty # penalty if goal is not completeed by the deadline
        self.non_goal = non_goal

    ### Get functions ###
    def getTasks(self):
        return self.tasks

    # return reward based on time
    def getReward(self, time):
        times = sorted(self.reward.keys())
        t = next(val for x, val in enumerate(times) if val >= time)
        return self.reward[t]

    # return deadline time, does not do any computation, just return self.deadline
    def getDeadline(self):
        return self.deadline

    def deadlinePenalty(self):
        return penalty

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

    def isNonGoal(self):
        return self.non_goal


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

    def __init__(self, todolist):
        self.todolist = todolist
        # self.state = (tasks, time)
        tasks = self.tasksToBinary(self.todolist.getTasks())
        time = self.todolist.getTime()

        # parameters
        self.livingReward = 0.0
        self.noise = 0.0

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
        binary_tasks = [1 if task.isComplete() else 0 for task in tasks]
        return binary_tasks

    def getState(self):
        return self.state

    def getStates(self):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        abstract

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        start_state = [0] * len(self.todolist.getTasks())
        return (start_state, 0)

    def getPossibleActions(self, state):
        """
        Return list of possible actions from 'state'.
        """
        tasks = state[0]
        possible_actions = [i for i, task in enumerate(tasks) if task == 0]
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
        task_index = self.tasks.index(action)

        binary_tasks = state[0]

        # state for not completing task
        next_states_probs.append((binary_tasks, state[1] + action.getTimeCost()), 1 - action.getProb())
        # state for completing task
        binary_tasks[task_index] = 1
        next_states_probs.append((binary_tasks, state[1] + action.getTimeCost()), action.getProb())

        return next_states_probs

    def getReward(self, state, action, nextState):
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        abstract

    def isTerminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        if self.time > self.end_time or len(self.incompleted_tasks) == 0:
            return True
        else:
            return False



if __name__ == '__main__':

    goals = [
        Goal("Goal A", [
            Task("Task A1", 1), 
            Task("Task A2", 1), 
            Task("Task A3", 1)], 
            {1: 14, 3: 10, 5: 8},
            penalty=-10),
        Goal("Goal B", [
            Task("Task B1", 1), 
            Task("Task B2", 2), 
            Task("Task B3", 2)], 
            {1: 10, 6: 5, 9: 1},
            penalty=-100),
        Goal("Goal C", [
            Task("Task C1", 1), 
            Task("Task C2", 2), 
            Task("Task C3", 3)], 
            {1: 100, 6: 90, 15: 80},
            penalty=-1000),
    ]

    my_list = ToDoList(goals, start_time=0, end_time=10)
    my_list.printDebug()

