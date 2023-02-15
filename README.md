# GridWorld
Grid world example in reinforcement learning

The following figure illustrate a recangular gridword representation as a finite MDP.

The cells of the grid are the states, the deterministic actions allow the agent go north, south, east or west.

The actions take the agent off the gird leave its location unchaged would get a reward of -1.
Other actions result in a reward of 0, except those that move the agent out of the special states A and B.
From state A, all four actions yield a reward of +10 and take the
agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'.

<img width="463" alt="image" src="https://user-images.githubusercontent.com/121702927/218357550-da633221-b507-4460-9bfd-01d2afa4b7c9.png">

To solve this problem. I reffer the matrix method proposed by this website(https://minibatchai.com/rl/2021/08/31/Bellman-1.html).
I believe the method proposed by the author is a good linear algebra practice.

In a MDP, The policy is a distribution over the actions a given a state. the value of a state s under policy $\pi$ is given by

<img width="649" alt="image" src="https://user-images.githubusercontent.com/121702927/218911672-8c119da7-e3ed-495e-918f-49a4efaa3e52.png">

The expected value of the reward from this state is given by

<img width="637" alt="image" src="https://user-images.githubusercontent.com/121702927/218911935-b43a4d1c-8997-4b98-8f68-ba8e5f6f7800.png">

The return at timestep t,Gp is deinfed as the discounted sum of all  future rewards starting at t:
<img width="661" alt="image" src="https://user-images.githubusercontent.com/121702927/218915966-d1d09d97-10d0-4761-a275-840029d3eb9b.png">

The expected return following state s at timestep t can be written as 

<img width="865" alt="image" src="https://user-images.githubusercontent.com/121702927/218916085-62dd76bb-6841-455f-b873-6d5ec54cf6ea.png">

Plugging in the expression for $v_\pi$ we get

<img width="644" alt="image" src="https://user-images.githubusercontent.com/121702927/218924065-e9a7dde9-7833-4e00-bad1-3263c9bc4e76.png">

Thus we can write the Bellman Equation

<img width="638" alt="image" src="https://user-images.githubusercontent.com/121702927/218924139-28d1008d-470a-4e08-90ed-54754edebbde.png">






