# GridWorld-
Grid world example in reinforcement learning

The following figure illustrate a recangular gridword representation as a finite MDP.

The cells of the grid are the states, the deterministic actions allow the agent go north, south, east or west.

The actions take the agent off the gird leave its location unchaged would get a reward of -1.
Other actions result in a reward of 0, except those that move the agent out of the special states A and B.
From state A, all four actions yield a reward of +10 and take the
agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'.

<img width="463" alt="image" src="https://user-images.githubusercontent.com/121702927/218357550-da633221-b507-4460-9bfd-01d2afa4b7c9.png">
