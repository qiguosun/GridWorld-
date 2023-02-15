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
I believe the method proposed by the author is a good linear algebra practice. The following is the theory part.

In a MDP, the value of a state s under policy $\pi$ is given by

**Value**
$$v_\pi (s) = E_\pi [G_t|S_t =s ] = E_{\pi} [\sum_{k=0}^\infty  \gamma ^k R_{t+k+1} |  S_t = s] \forall s \in S $$

The policy is a distribution over the actions $a$ given a state $s$. The expected value of the reward from this state is given by

**Expected value of the reward**
$$E_\pi [R_{t+1}|S_t = s]=\sum_r r\cdot p(R_{t+1}=r|S_t=s) = \sum_{s',r,a} r\cdot p(R_{t+1}=r,S_{t+1}=s'|A_t=a,S_t=s)\cdot \pi(A_t=a|S_t=s)   $$

The return at timestep t,$G_p$ should be the discounted sum of future rewards starting at t, given by

**Expected return**
$$E_\pi[G_{t+1}|S_t=s] = \sum_{s',a}E_\pi[G_{t+1}|S_{t+1}=s',S_t=s]\cdot p(S_{t+1}=s'|S_t=s,A_t=a)\cdot\pi(A_t=a|S_t=s)=\sum_{s',a}E_\pi[G_{t+1}|S_{t+1}=s'] p(S_{t+1}=s'|S_t=s,A_t=a)\cdot\pi(A_t=a|S_t=s)$$

The expected return following state s at timestep t can be written as 

$$E_\pi[G_{t+1}|S_t=s] = \sum_{s',a}E_\pi[G_{t+1}|S_{t+1}=s',S_t=s]\cdot p(S_{t+1}=s'|S_t=s,A_t=a)\cdot\pi(A_t=a|S_t=s)=\sum_{s',a}E_\pi[G_{t+1}|S_{t+1}=s'] p(S_{t+1}=s'|S_t=s,A_t=a)\cdot\pi(A_t=a|S_t=s)$$

Then we plugging in the expression for $v_\pi$ defined in the first equation

$$E_\pi[G_{t+1}|S_t=s] = \sum_{s',r,a} v_\pi(s')\cdot p(R_{t+1}=r,S_{t+1}=s'|S_t=s,A_t=a)\cdot \pi(A_t=a|S_t=s)$$

Thus the Bellman Equation can be expressed as

**Bellman equation**
$$v_\pi(s)=E_\pi [R_{t+1}+ \gamma G_{t+1}|S_t=s] = \sum_a \pi(a|s) \sum_{s'}\sum_r p(s',r|s,a)[r+\gamma v_\pi (s')]$$

Then the auther proposed vectorised form for calculation. For N discrete reward values $\rho_0,...\rho_N$, let be $\boldsymbol{\rho}$ a column vector where $\boldsymbol{\rho_r} =r$.Then assuming that $s=0,1,...,|S|-1$, i.e., mapping states to indexes

$$\sum_r r \sum_{a,s'} p(s',r|s,a)\cdot \pi(a|s)=\sum_r \boldsymbol{\rho_r} \sum_{a,s'} \pi(a|s)\cdot p(s',r|s,a) = (\boldsymbol{Q}\boldsymbol{\rho})$$


where  
$$\boldsymbol{Q}\in R^{N\times |S|}, Q_{rs}=\sum_{a,s'} p(s',r|s,a) \cdot \pi(a|s) $$

with $v$ as a row vector with dimension $|S|$, where $\boldsymbol{v}_{s'}=v_\pi (s')$

$$\sum_a \pi(a|s)\sum_{s',r} \gamma v_\pi(s')\cdot p(s',r|s,a) = \gamma \sum_{s'} \boldsymbol{v}_{s'} \cdot \sum_{a,r} p(s',r|s,a)\cdot \pi(a|s) = \gamma(\boldsymbol{Rv})_{s} $$

where 

$$\boldsymbol{R}\in R^{|S|\times|S|}, \boldsymbol{R}_{s',s}=\sum_{a,r}p(s',r|s,a)\cdot \pi(a|s)$$

write the Bellman equation in vectorised form

$$\boldsymbol{v} = \boldsymbol{Q}\boldsymbol{\rho}+\gamma\boldsymbol{Rv}$$

Solving for $\boldsymbol{v}$:

$$(\boldsymbol{I}-\gamma\boldsymbol{R})\boldsymbol{v} = \boldsymbol{Q}\boldsymbol{\rho} \implies \boldsymbol{v}=(\boldsymbol{I}-\gamma\boldsymbol{R})^{-1} \boldsymbol{Q\rho}$$


