SARSA Reinforcement Learning in a Grid World
This project implements the State-Action-Reward-State-Action (SARSA) reinforcement learning algorithm to train an intelligent agent (robot) to navigate a grid-based environment. The goal of the robot is to find the optimal path from a starting point to a target destination while avoiding obstacles.

Table of Contents
Introduction
Features
Environment Setup
How It Works
Code Structure
Configuration
Usage
SARSA Algorithm Briefly
Introduction
Reinforcement Learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize the notion of cumulative reward. SARSA is an on-policy temporal difference learning algorithm that learns the optimal policy. In this project, we apply SARSA to a simple grid world where the agent learns to move around, avoid obstacles, and reach a target.

Features
Customizable Grid: 

Easily adjust the grid size, number of obstacles, and reward values.
SARSA Implementation: A clear and functional implementation of the SARSA algorithm.
Epsilon-Greedy Exploration: Balances exploration (trying new actions) and exploitation (using learned knowledge).
Dynamic Obstacle Placement: Obstacles are randomly placed, ensuring varied training scenarios.
Console Visualization: Provides a basic console-based visualization of the robot's movement and the environment during testing.
Environment Setup
To run this project, you need Python 3 and the numpy library.

Clone the repository (or save the code):

```bash
git clone <repository_url> # If it's in a repo

Or simply save the provided Python code as sarsa_grid_world.py.
```
Install dependencies:
```bash

pip install numpy
```
How It Works
The program first initializes a grid environment with a start state, a target state, and randomly placed obstacles. It then trains a Q-table using the SARSA algorithm over a specified number of episodes. During training, the robot explores the environment and updates its knowledge about the value of taking certain actions in specific states. After training, the learned policy (represented by the Q-table) is tested, and the robot attempts to find the optimal path to the target.

Code Structure
GRID_SIZE, START_STATE, TARGET_STATE, NUM_OBSTACLES: Define the environment's basic parameters.

ACTIONS: A dictionary mapping action IDs to their names (UP, DOWN, LEFT, RIGHT).

REWARD_GOAL, REWARD_OBSTACLE, REWARD_MOVE: Reward values for different outcomes.

ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, EPISODES: Hyperparameters for the SARSA algorithm.

create_grid_and_obstacles(): Generates the grid and places obstacles.

get_state_index(), get_state_from_index(): Utility functions to convert between state coordinates and a single index for the Q-table.

get_next_state_reward_done(): Determines the next state, reward, and if the episode is finished based on an action.

choose_action(): Implements the epsilon-greedy policy for action selection.

render_grid(): Visualizes the current state of the grid in the console.

train_sarsa(): The main function for training the SARSA agent.

test_sarsa_policy(): Tests the learned policy by letting the robot navigate the grid.

Configuration
You can modify the following constants at the beginning of the script to experiment with different scenarios:

GRID_SIZE: The dimensions of the square grid (e.g., 5 for a 5x5 grid).

START_STATE: The starting coordinates of the robot (e.g., (0, 0)).

TARGET_STATE: The target coordinates the robot aims to reach.

NUM_OBSTACLES: The number of randomly placed obstacles.

REWARD_GOAL: Reward for reaching the target.

REWARD_OBSTACLE: Penalty for hitting an obstacle or a wall.

REWARD_MOVE: Penalty for each normal movement.

ALPHA: Learning Rate (how much new information overrides old information).

GAMMA: Discount Factor (importance of future rewards).

EPSILON_START, EPSILON_END, EPSILON_DECAY: Parameters controlling the exploration rate over time.

EPISODES: The total number of training episodes.

Usage
To run the simulation, simply execute the Python script:


```bash
python sarsa_grid_world.py
```

The script will first perform the SARSA training, printing a message when training is complete. Afterward, it will automatically switch to policy testing mode, visualizing the robot's movement in the grid world based on the learned Q-table.

During the test phase, you will see the grid updated in your console, showing:

R: Robot's current position
X: Obstacles
T: Target
*: Path taken by the robot
SARSA Algorithm Briefly
SARSA is an on-policy TD (Temporal Difference) control algorithm. This means it learns the value function of the policy that is currently being followed, including the exploration steps. The update rule for the Q-value of a state-action pair (S,A) is:

Q(S,A)←Q(S,A)+α[R+γQ(S 
′
 ,A 
′
 )−Q(S,A)]

Where:

Q(S,A) is the Q-value of the current state S and action A.
alpha is the learning rate.
R is the immediate reward received after taking action A in state S.
gamma is the discount factor.
S 
′
  is the next state.
A 
′
  is the next action, chosen using the same policy (e.g., epsilon-greedy) that chose A. This is the key difference from Q-learning, which uses the maximum Q-value for the next state.
