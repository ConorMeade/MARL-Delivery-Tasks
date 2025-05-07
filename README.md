## MARL - Delivery Tasks

Conor Meade

CS138 - Reinfocement Learning

This project is an examination into Multi Agent Reinforcement Learning. The PettingZoo simple_spread_v3 environment is used to fascilitate these experiments.

Given some number of agents (default of 3), agents will be tasked with reaching all of the pickup locations within some number of episodes (usually in the 30-70 range). 

An actor approach will be used for determining actions via a neural network. A critic will be used for evaluating states for the value function. Batch updates will be made after a batch of 16 rollouts is reached. With this batch update, we compute our loss values and use the Adam optimizer for gradient descent and then back propogating to update network values.


#### Installing Dependencies and Running the Code

To install the needed packages, enter:

`pip install -r requirements.txt`

To do a single-seeded run, enter

`python main.py` or `python3 main.py` 

Would depend on how python path is configured

Similarly, to run across five different seeds (for computing mean and std dev), enter:

`python main_seeds.py` or `python3 main_seeds.py`
