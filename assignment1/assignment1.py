from matplotlib import pyplot as plt
from gridWorld import gridWorld
import numpy as np
from copy import deepcopy


def show_value_function(mdp, V):
    fig = mdp.render(show_state = False, show_reward = False)            
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        fig.axes[0].annotate("{0:.3f}".format(V[k]), (s[1] - 0.1, s[0] + 0.1), size = 40/mdp.board_mask.shape[0])
    plt.show()
    
def show_policy(mdp, PI):
    fig = mdp.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in mdp.states():
        s = k if isinstance(k, tuple) else mdp.legal_states[k]
        if mdp.terminal[s] == 0:
            fig.axes[0].annotate(action_map[PI[k]], (s[1] - 0.1, s[0] + 0.1), size = 100/mdp.board_mask.shape[0])
    plt.show()
    
####################  Problem 1: Value Iteration #################### 

def value_iteration(mdp, gamma, theta = 1e-3):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    """
    YOUR CODE HERE:
    Problem 1a) Implement Value Iteration
    
    Input arguments:
        - mdp     Is the markov decision process, it has some usefull functions given below
        - gamma   Is the discount rate
        - theta   Is a small threshold for determining accuracy of estimation
    
    Some usefull functions of the grid world mdp:
        - mdp.states() returns a list of all states [0, 1, 2, ...]
        - mdp.actions(state) returns list of actions ["U", "D", "L", "R"] if state non-terminal, [] if terminal
        - mdp.transition_probability(s, a, s_next) returns the probability p(s_next | s, a)
        - mdp.reward(state) returns the reward of the state R(s)
    """

    while True:
        delta = 0
        for i in range(len(mdp.states())):
            s = mdp.states()[i]
            v = V[i]

            if len(mdp.actions(s)) == 0:
                V[i] = mdp.reward(s) 
            else:
                V[i] = max([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])
            
            delta = max(delta, abs(v - V[i]))
        if delta < theta:
            break

    return V

def policy(mdp, V):
    # Initialize the policy list of crrect length
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 1b) Implement Policy function 
    
    Input arguments:
        - mdp Is the markov decision problem
        - V   Is the optimal falue function, found with value iteration
    """
    for i in range(len(mdp.states())):
        s = mdp.states()[i]

        if len(mdp.actions(s)) == 0:
            PI[i] = 0

        else:
            best_action_idx = np.argmax([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])
            PI[i] = mdp.actions(s)[best_action_idx]
    
    return PI

####################  Problem 2: Policy Iteration #################### 
def iterative_policy_evaluation(mdp, gamma, PI, V, theta = 1e-3):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        - theta Is small threshold for determining accuracy of estimation
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    V = np.zeros((len(mdp.states())))
    while True:
        delta = 0
        for i in range(len(mdp.states())):
            s = mdp.states()[i]
            v = V[i]
            if len(mdp.actions(s)) == 0:
                V[i] = mdp.reward(s)
            else:
                V[i] = sum(mdp.transition_probability(s, PI[s], s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states())
            delta = max(delta, abs(v - V[i]))
        if delta < theta:
            break

    print(V)
    return V

def explicit_policy_evaluation(mdp, gamma, PI, V):   
    """
    YOUR CODE HERE:
    Problem 2a) Implement Policy Evaluation
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor
        - PI    Is current policy
        - V     Is preveous value function guess
        
    Some useful tips:
        - If you decide to do exact policy evaluation, np.linalg.solve(A, b) can be used
          optionally scipy has a sparse linear solver that can be used
        - If you decide to do exact policy evaluation, note that the b vector simplifies
          since the reward R(s', s, a) is only dependant on the current state s, giving the 
          simplified reward R(s) 
    """
    n = len(mdp.states())
    V = np.zeros(n)

    A = np.zeros(shape=(n, n))
    b = np.zeros(n)
    for i in range(n):
        s = mdp.states()[i]
        A[i] = [gamma*mdp.transition_probability(s, PI[i], s_next) for s_next in mdp.states()]
        if len(mdp.actions(s)) == 0:
            b[i] = -mdp.reward(s)
        else: 
            b[i] = -sum(mdp.transition_probability(s, PI[i], s_next)*mdp.reward(s) for s_next in mdp.states())
    A = A - np.identity(n)

    V = np.linalg.solve(A, b) # prone to singular matrix. Works if you try to re-run the program a couple of times or decrease gamma

    return V

def policy_iteration(mdp, gamma, implicit_eval = True):
    # Make a valuefunction, initialized to 0
    V = np.zeros((len(mdp.states())))
    
    # Create an arbitrary policy PI
    PI = np.random.choice(env.actions(), len(mdp.states()))
    
    """
    YOUR CODE HERE:
    Problem 2b) Implement Policy Iteration
    
    Input arguments:  
        - mdp   Is the markov decision problem
        - gamma Is discount factor

    Some useful tips:
        - Use the the policy_evaluation function from the preveous subproblem
    """
    while True:
        PI_old = deepcopy(PI)
        V = iterative_policy_evaluation(mdp, gamma, PI, V) if implicit_eval else explicit_policy_evaluation(mdp, gamma, PI, V)
        
        for i in range(len(mdp.states())):
            s = mdp.states()[i]
            if len(mdp.actions(s)) == 0:
                PI[i] = 0
            else:
                best_policy_idx = np.argmax([sum(mdp.transition_probability(s, a, s_next) * (mdp.reward(s) + gamma*V[s_next]) for s_next in mdp.states()) for a in mdp.actions(s)])
                PI[i] = mdp.actions(s)[best_policy_idx]
        if np.array_equal(PI, PI_old):
            break

    return PI, V

if __name__ == "__main__":
    """
    Change the parameters below to change the behaveour, and map of the gridworld.
    gamma is the discount rate, while filename is the path to gridworld map. Note that
    this code has been written for python 3.x, and requiers the numpy and matplotlib
    packages

    Available maps are:
        - gridworlds/tiny.json
        - gridworlds/large.json
    """
    gamma   = 0.9
    filname = "gridworlds/large.json"

    # Import the environment from file
    env = gridWorld(filname)

    # Render image
    fig = env.render(show_state = False)
    plt.show()

    # Run Value Iteration and render value function and policy
    V = value_iteration(mdp = env, gamma = gamma)
    show_value_function(env, V)

    PI = policy(env, V)
    show_policy(env, PI)

    # Run Policy Iteration and render value function and policy
    PI, V = policy_iteration(mdp = env, gamma = gamma, implicit_eval=False)
    show_value_function(env, V)
    show_policy(env, PI)
