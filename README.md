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

# Optimal value function

**Belman optimality equation**
$$v_{\*}(s) = \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma v^{\*}(s')]$$

Let $P_{s'rsa}$ be a 4D tensor and $P_{s':sa}$ be where the |R|-d vector obtained by fixting  s',s and a whilst $P_{:rsa}$ be the |S|-d vector obtained by fixing r,s,and a.

$$v^{\*}_s = \max_a(\sum_{s'} \boldsymbol{P}_{s':sa} \boldsymbol{\rho} + \gamma \sum_r \boldsymbol{P}^T_{:rsa} v^{\*})$$

## Estimating the best poliby $\pi^{\*}$ iteratively
This equation for $v^{\*}_s$ is non-linear equation which needs to be solved iteratively. we will initialise v and $\pi$ with some value and updating v with the max over all actions a as illustrated in the function for $v^{\*}_s$ until the difference becomes negligible. The actions at a given state that yield the best value will give us the best policy $\pi^{\*}$.

The method of Iterative Policy Evaluation given in Sutton and Barto is as follows

<img width="626" alt="image" src="https://user-images.githubusercontent.com/121702927/219266011-f21ddea0-4e58-4076-a8fc-5aaf7531ca20.png">

A vectorised approach (https://minibatchai.com/rl/2021/08/31/Bellman-1.html) is given by 

```python
def estimate_vstar(v_init, prob_next_reward, rewards, gamma = 0.9, tol=1e-12):
    iters = 0
    v_prev = v_init
    vs = [v_prev]
    diffs = []
    while True:
        iters+=1
        QQ = np.einsum('Srsa,r->sa', prob_next_reward, rewards)
        RR = np.einsum('Srsa,S->sa',prob_next_reward, v_prev)
        v_next = np.max(QQ + gamma*RR, axis=-1)
        diff = np.square(v_prev-v_next)
        diffs.append(np.mean(diff))
        if iters % 20 == 0:
            print('\rIteration {}, mean squared difference {}'.format(iters, diffs[-1]))
        vs.append(v_next)
        if np.all(diff < tol):
            break
        v_prev = v_next 
        
    print('\rFinal iteration {}, mean squared difference {}'.format(iters, diffs[-1]))
    return v_next, vs, diffs, QQ, RR
```

Then it's time to solve the problem defined in Example 3.8. The initiall states were set to 0 and the action is considered equally good. In addition, the same environment allows same P tensor. 

```python
vstar, vs, diffs, QQ, RR = estimate_vstar(np.zeros((5,5)), P, rewards)
```
