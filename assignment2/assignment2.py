from matplotlib import pyplot as plt
from gridWorld import gridWorld
from pendulum import pendulum
import numpy as np
import math

def show_action_value_function(env, Q):
    pos = {"U": (-0.15, -0.3), "D": (-0.15, 0.4), "L": (-0.45, 0.1), "R": (0.05, 0.1)}
    fig = env.render(show_state = False, show_reward = False)            
    for k in env.states():
        s = env.legal_states[k]
        for i, a in enumerate(env.actions(k)):
            fig.axes[0].annotate("{0:.2f}".format(Q[k, i]), (s[1] + pos[a][0], s[0] + pos[a][1]), size = 40/env.board_mask.shape[0], color = "r" if Q[k, i] == max(Q[k, :]) else "k")
    plt.show()
    
def show_policy(env, Q):
    fig = env.render(show_state = False, show_reward = False)
    action_map = {"U": "↑", "D": "↓", "L": "←", "R": "→"}
    for k in env.states():
        s = k if isinstance(k, tuple) else env.legal_states[k]
        if not env.terminal(s):
            fig.axes[0].annotate(action_map[env.actions(s)[np.argmax(Q[k, :])]], (s[1] - 0.1, s[0] + 0.1), size = 100/env.board_mask.shape[0])
    plt.show()


####################  Problem 1: Q-Learning #################### 
def Q_Learning(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()

    """
    YOUR CODE HERE:
    Problem 1a) Implement Q-Learning
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """
    
    while (not done):
        a = np.argmax(Q[s]) if np.random.random_sample() <= epsilon else np.random.randint(low=0, high=Q.shape[1])
        action = env.actions()[a]
        s_next, r, done = env.step(action)
        Q[s][a] += alpha * (r + gamma * max(Q[s_next]) - Q[s][a])
        s = s_next
    
    return Q

####################  Problem 2: SARSA #################### 
def SARSA(env, gamma, Q, alpha, epsilon):
    # Reset environment
    s, r, done = env.reset()

    """
    YOUR CODE HERE:
    Problem 2a) Implement SARSA
    
    Input arguments:
        - env     Is the environment
        - gamma   Is the discount rate
        - Q       Is the Q table
        - alpha   Is the learning rate
        - epsilon Is the probability of choosing greedy action
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """
    a = np.argmax(Q[s]) if np.random.random_sample() <= epsilon else np.random.randint(low=0, high=Q.shape[1])

    while (not done):
        action = env.actions()[a]

        s_next, r, done = env.step(action)
        
        if len(env.actions()) == 0:
            a_next = 0
        else:
            a_next = np.argmax(Q[s_next]) if np.random.random_sample() <= epsilon else np.random.randint(low=0, high=Q.shape[1])

        Q[s][a] += alpha * (r + gamma * Q[s_next][a_next] - Q[s][a])

        s, a = s_next, a_next
    
    return Q


def convert_state_to_indices(s, n_theta, n_theta_dot):
    for i in range(1, n_theta + 1):
        if s[0] <= -math.pi + i * (2 * math.pi) / n_theta:
            s1 = i - 1
            break

    for i in range(1, n_theta_dot + 1):
        if s[1] <= -10 + i * 20 / n_theta_dot:
            s2 = i - 1
            break

    return s1, s2


def Q_Learning_pendulum(env, gamma, Q, alpha, epsilon, n_theta, n_theta_dot):
    """
    Input arguments:
        - env           Is the environment
        - gamma         Is the discount rate
        - Q             Is the Q table
        - alpha         Is the learning rate
        - epsilon       Is the probability of choosing greedy action
        - n_theta       Is the number of discretized states for the continuous theta state
        - n_theta_dot   Is the number of discretized states for the continuous theta_dot state
    
    Some usefull functions of the grid world environment
        - s_next, r, done = env.step(a)  Take action a and observe the next state, reward and environment termination
        - actions = env.actions()        List available actions in current state (is empty if state is terminal)
    """
    # Reset environment
    s, r, done = env.reset()
    s1, s2 = convert_state_to_indices(s, n_theta, n_theta_dot)
    
    while (not done):
        a = np.argmax(Q[s1][s2]) if np.random.random_sample() <= epsilon else np.random.randint(low=0, high=Q.shape[2])
        action = env.actions()[a]

        s_next, r, done = env.step(action)
        s1_next, s2_next = convert_state_to_indices(s_next, n_theta, n_theta_dot)

        Q[s1][s2][a] += alpha * (r + gamma * max(Q[s1_next][s2_next]) - Q[s1][s2][a])
        s1, s2 = s1_next, s2_next
    
    return Q


if __name__ == "__main__":
    """
    Note that this code has been written for python 3.x, and requiers the numpy, matplotlib
    and scipy packages.
    """

    # Import the environment from file
    filename = "gridworlds/tiny.json"
    env = gridWorld(filename)

    # Render image
    fig = env.render(show_state = True)
    plt.show()
    
    """
    Problem 1 (Run Q-Learning)
    
    Below is the code for running Q-Learning, feel free to change the code, and tweek the parameters.
    """
    gamma = 1.0     # Discount rate
    alpha = 0.1     # Learning rate
    epsilon = 0.9   # Probability of taking greedy action
    episodes = 5000 # Number of episodes
    
    Q = np.zeros([len(env.states()), 4])
    for i in range(episodes):
        Q_Learning(env, gamma, Q, alpha, epsilon)

    # Render Q-values and policy 
    show_action_value_function(env, Q)
    show_policy(env, Q)

    """
    Problem 2 (Run SARSA)
    
    Below is the code for running SARSA, feel free to change the code, and tweek the parameters.
    """
    gamma = 1.0     # Discount rate
    alpha = 0.1     # Learning rate
    epsilon = 0.9   # Probability of taking greedy action
    episodes = 5000 # Number of episodes

    Q = np.zeros([len(env.states()), 4])
    for i in range(episodes):
        SARSA(env, gamma, Q, alpha, epsilon)

    # Render Q-values and policy 
    show_action_value_function(env, Q)
    show_policy(env, Q)

    """
    Problem 3) Pendulum 
    
    For this problem the aim is to use either Q-Learning or SARSA for the pendulum environment. You should be able to
    reuse most of the Q-Learning and SARSA algorithms form the preveous problems, but you may have to make two main 
    changes:
        
        1)  The states of the pendulum environment are continous, you must crteate a discretization of the state so 
            that it can be stored in a Q-table. 
    
        2)  The state of the pendulum has multiple elements (s = [theta, theta_dot]) you must change the code so that
            Q-table is able to store this. The simplest way ofdoing this is to create a three dimensional Q-Table in 
            the following way:  Q[theta, theta_dot, action]
    """

    # Create instance of pendulum environment
    env = pendulum()
    fig = env.render()

    gamma   = 0.99  # Discount rate
    alpha   = 0.2   # Learning rate
    epsilon = 0.5   # Probability of taking greedy action
    episodes = 5000 # Number of episodes

    n_theta = 100       # number of discretized states for theta state
    n_theta_dot = 100    # number of discretized states for theta_dot state


    Q = np.zeros([n_theta, n_theta_dot, 3])
    for i in range(episodes):
        print(f'Episode: {i}')
        Q_Learning_pendulum(env, gamma, Q, alpha, epsilon, n_theta, n_theta_dot)


    # Plot the value function
    V = np.max(Q, axis=2)   # Value function is given as V(s) = max_a Q(s, a) 
    V[V==0] = np.nan        # Assume states with zeros are not visited and set them to NAN (gives better plot)

    plt.figure()
    plt.imshow(V, cmap="viridis")
    plt.colorbar()
    plt.show()


    # Run greedy policy for 10 second to see how the trained policy behaves
    plt.ion()
    fig = env.render()
    s, _, _ = env.reset([np.pi, 0])
    for i in range(200):
        s1, s2 = convert_state_to_indices(s, n_theta, n_theta_dot)
        a = np.argmax(Q[s1][s2][:])
        s, _, _ = env.step(a)
        plt.pause(env.step_size)