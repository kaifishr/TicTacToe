# TicTacToe 

$$
\begin{aligned}
&\begin{array}{c|c|c}
\times & \circ & \circ \\
\hline
\circ & \times &  \circ \\
\hline
\circ & \times & \times \\
\end{array}
\end{aligned}
$$

A minimal environment equipped with reinforcement learning algorithms to train agents to play [Tic-tac-toe](https://en.wikipedia.org/wiki/Tic-tac-toe). Due to its simplicity, this repository is potentially useful for educational purposes and can serve as a starting point to solve other games such as a generalization of m,n,k-games, chess or Go.


## Introduction

Tic-tac-toe is an instance of a perfect information, turn-based two player [**m,n,k-game**]( https://en.wikipedia.org/wiki/M,n,k-game) and is also called a **k-in-a-row** game on an **m-by-n** board. For Tic-tac-toe, these variables are $m=n=k=3$.

In this implementation, two agents alternate taking turns on an $m \times n$ board until one of them gets $k$ in a row and wins the game. The current implementations supports any positive integer for $m,n,k$ with $m=n=k$.


## Reinforcement Learning

This section is intended to give a very brief introduction to some aspects of reinforcement learning and the algorithms, [Policy Gradients](#policy-gradients) and [Deep Q-Learning](#deep-q-learning), that are used to train the agents to play Tic-tac-toe.

If the learning task that we try to solve can not be taken offline (because we already are in the possession of labeled training data) and reduced to an abstract task like regression or classification we can formulate it as an reinforcement learning task.

In reinforcement learning, agents interact with an environment, perform actions, and are continually trained to learn how to correctly interact with a dynamic world.

To be more specific, let's use the game of Tic-tac-toe as an example. The dynamic world or environment is represented by the Tic-tac-toe game.

An agent ($x$) observes a **state** that is represented by the current board configuration (positions of $\times$ and $\circ$). An example state can look as follows:

$$
\begin{aligned}
&\begin{array}{c|c|c}
\circ &  & \circ \\
\hline
& \times &  \\
\hline
\circ & \times & \times \\
\end{array}
\end{aligned}
$$

Based on the observed state, the agent performs an **action**. This action causes the environment to transition to a new state. Available actions is the set of allowed moves.

Followed by the action a **reward** is provided by the environment. The reward is a scalar value, where higher values are better.

The agent's action is based on a **policy**. A policy is a function that maps states to actions and can be modeled by a neural network whose parameters $\theta$ are learned.

$$\text{action}= \text{policy}(\text{state}; \theta)$$

During the training, the agent interacts with the environment and the selected optimization method, adjusts the agent's policy in order to *maximize the expectation of future rewards*.

It should be noted, that for a given state, the agent's best choice for an action depends only on the current state and may not depend on past states. Considering only information provided by the current state for the next action is known as a [Markow decision process](https://en.wikipedia.org/wiki/Markov_decision_process) (MDP).


### Tic-tac-toe Environment

This section describes the details of the Tic-tac-toe environment as well as the action and reward scheme.

The state that the agent observes are the positions of crosses ($\times = 1$), circles ($\circ = -1$, the opponent), and empty fields ($0$) on the board.

$$
\begin{aligned}
&\begin{array}{c|c|c}
\circ &  & \circ \\
\hline
& \times &  \\
\hline
\circ & \times & \times \\
\end{array}
\end{aligned} =
\begin{aligned}
&\begin{array}{c|c|c}
-1 & 0 & -1 \\
\hline
0 & 1 & 0 \\
\hline
-1 & 1 & 1 \\
\end{array}
\end{aligned}
$$

Based on that state, the agent selects one out of nine actions. Actions are determined by the agent's current policy. Here, the policy is modeled as a neural network with nine output neurons. The actions are integer-valued and retrieved by applying the argmax function to the network's output neurons.

Even though there are nine actions available, not all moves are allowed. If the action is legal, the opponent makes a move and the agent observes the new state.

States where the game is won come with a reward of +1. Loosing the game or marking already occupied areas results in a reward of -1. Punishing wrong moves encourages the agent to learn only legal moves over time. All other states (including draws) yield a reward of 0. 


### Policy Network

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) showed that deep neural networks are a powerful option to represent reinforcement learning models that map states to (a distribution over) actions.

The policy network receives a state vector $s$ holding nine numbers ($-1$, $0$, or $1$) and uses a softmax output layer to return a probability distribution over the nine possible actions. Using the example from above we can illustrate this as follows: 

$$
\begin{pmatrix}
0.0 & 0.8 & 0.0\\
0.1 & 0.0 & 0.1\\
0.0 & 0.0 & 0.0
\end{pmatrix}
= \text{policy}
\begin{pmatrix}
\begin{pmatrix}
-1 & 0 & -1 \\
0 & 1 & 0 \\
-1 & 1 & 1
\end{pmatrix}
; \theta
\end{pmatrix}
$$

We can choose an action by either choosing the action with the highest probability or by sampling from the output probability distribution.


## Episodic Learning

In a reinforcement learning setting, an agent can theoretically learn a task in an online mode ([see this example](https://arxiv.org/pdf/2208.07860.pdf)), where the agent's policy (the neural network) is continuously updated. However, in practice, this can lead to unpredictable behavior of the agent that is difficult to control.

Instead of updating the agent's policy at every time step, a common approach is to update the policy between episodes. An episode can be defined as a task we want the agent to learn. For this project, one episode is a game of Tic-tac-toe, but it can also be the task of [landing a rocket booster autonomously](https://github.com/kaifishr/RocketLander).

During an episode, the agent takes actions according to its current policy and collects the rewards. We then use this information to update the policy's parameters and start a new episode.


## Self-play

This framework allows to train agents using a self-play training strategy. Instead of an algorithmic player as opponent, we train two agents at the same time and let them play against themselves. Alternatively, we can train an agent several episodes until it beats a weaker version of itself for number of times. We then make the trained agent the new opponent and start over again. 

TODO To ensure that the agent generalizes well, it is a good strategy to have an ensemble of opponent agents and sample one at random for each episode. 


### (TODO) Policy Gradients

The policy is a neural network that outputs probabilities over `size**2` possible actions. Thus, we can interpret the resulting probabilities as a probabilistic policy.
...


### (TODO) Deep Q-Learning

...



## References

* [A Crash Course on Reinforcement Learning](https://arxiv.org/abs/2103.04910)

* [Lecture 13: Reinforcement learning](https://mlvu.github.io/lecture13/)


## TODO

- Add more theory to readme.
- Generalize implementation for m,n,k-games.
- Implement adaptive epsilon decay rate.

## License

MIT
