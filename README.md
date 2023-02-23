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

Tic-tac-toe is an instance of a perfect information, turn-based two player [**m,n,k-game**]( https://en.wikipedia.org/wiki/M,n,k-game) and is also called a **k-in-a-row** game on an **m-by-n** board. For Tic-tac-toe, $m=n=k=3$.

In this implementation, two agents alternate taking turns on an $m \times n$ board until one of them gets $k$ in a row and wins the game. The current implementations supports any positive integer for $m,n,k$ with $m=n=k$.


## Reinforcement Learning

This section is intended to give a very brief introduction to some aspects of reinforcement learning and the algorithms, [Policy Gradients](#policy-gradients) and [Deep Q-Learning](#deep-q-learning), that are used to train the agents to play Tic-tac-toe.

If the learning task that we try to solve can not be taken offline (because we already are in the possession of labeled training data) and reduced to an abstract task like regression or classification we can formulate it as an reinforcement learning task.

In reinforcement learning, agents interact with an environment, perform actions, and are continually trained to learn how to correctly interact with a dynamic world.

To be more specific, let's use the game of Tic-tac-toe as an example. The dynamic world or environment is represented by the Tic-tac-toe game.

An agent ($x$) observes a **state** that is represented by the current board configuration (positions of $\times$ and $\circ$). An example state can look as follows:

```
o . o
. x .
o x x
```
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

It should be noted, that for a given state, the agent's best choice for an action depends only on the current state and may not depend on past states. Considering only information provided by the current state for the next action is known as a **Markow decision process** (MDP) 

---

During the training, the agent interacts with its environment, and the learner, the selected optimization method, adjusts the policy in order to *maximize the expectation of future rewards*.

The policy is a neural network that outputs probabilities over `size**2` possible actions. Thus, we can interpret the resulting probabilities as a probabilistic policy.


---
---
---
---


### Tic-tac-toe

...


### Policy Gradients

...


### Deep Q-Learning

...


Available actions are the moves the player is allowed to make.

After the player's (agent's) action, the opponent makes a move, and the environment returns the resulting game state to the agent.

States where the game is won come with a reward of +1, and -1 if the game is lost. All other states (including draws) yield a reward of 0.

In short:

Action: The agent's move. Placing a cross or cicle (-1, 1).
State: State of playing field after opponent's move.
Reward: 1 (won), -1 (lost), 0 (otherwise).

To encourage the agent to make legal moves, illegal moves (marking already occupied areas) instantly result in the loss of the game. 

This framework allows to train agents using a self-play training strategy.

The agents are being trained *while* they are interacting with a dynamic world or environment. 



## Episodic Learning

- Train for one episode, observe reward, learn, repeat.
- Here, one episode is a game of Tic-Tac-Toe.
- This procedure results in a fixed policy for production.

As an aside, RL agents can theoretically learn in an online mode, where they continuously update their model while they explore their environment. Aside end.


## Self-play

Instead of an algorithmic player, we train two agents and let them play against themselves. Alternatively, we train an agent until it beats another randomly initialized player. We then make the trained agent the new opponent and start again.

To ensure that the agent generalizes well, it is a good strategy to have an ensemble of opponent agents and sample one at random for each episode. 


## References


## TODO

- Policy gradients
- Deep-Q
- Evolutionary strategies
- Generalize implementation for m,n,k-games.


## License

MIT
