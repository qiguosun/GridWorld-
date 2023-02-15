import numpy as np
import matplotlib.pyplot as plt
# global variables
h = w = 5
gamma = 0.9

# create states space with the shape of [25,2]
x, y = np.arange(h),np.arange(y)
xv, yv = np.meshgrid(x,y)
states = np.stack(xv,yv,axis=-1).reshape((25,2))
# to state-index pair form {(0,0):0,(0,1):1...}
states2ind = dict(zip(map(tuple, states), range(len(states))))

actions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
pi = np.tile([[0.25]], [4, 25])

state_a = np.array([0, 1])
state_b = np.array([0, 3])
next_a = np.array([4, 1])
next_b = np.array([2, 3])

rewards = np.array([-1, 0, 5, 10])

rewards2ind = dict(zip(rewards, range(len(rewards))))

# apply np.any on [True, Flase] -> True 
def out_of_bounds(state, h, w):
    return np.logical_or(np.any(state >= [h, w], axis=-1),
                         np.any(state < [0, 0], axis=-1))

  
P = np.zeros((states.shape[0], rewards.shape[0],
             states.shape[0], actions.shape[0]))

# Basically, given the present state s and action a, there is a single posible next state and reward,
# i.e., the posibility of s',r, p(s',r|s,a) =1. So here is a 4D tensor for p(s',r|s,a) shaped [25,4,25,4]
# For each s,a the matrix consists of a |S|x|R| tensor that is 1 ath the lement correspondingt to s',r' and 0 everywhere else


for s, state in enumerate(states):
    for a, action in enumerate(actions):
        if np.all(state == state_a):
            next_state = next_a
            reward = 10
        elif np.all(state == state_b):
            next_state = next_b
            reward = 5
        else:
            next_state = state + action
            if out_of_bounds(next_state, h, w):
                next_state = state
                reward = -1
            else:
                reward = 0
        P[states2ind[tuple(next_state)], rewards2ind[reward], s, a] = 1

# note that N is the reward space, Q -> R^{Nx|S|}
# R -> R^{|S|x|S|}

Q = np.einsum('Srsa,as->sr', P, pi)
R = np.einsum('Srsa,as->sS', P, pi)

y = np.einsum('sr,r->s', Q, rewards)
v = np.einsum('sS,S->s', np.linalg.inv(np.eye(R.shape[0])-gamma*R), y)


def make_grid(v, shape):
    return v.reshape(shape)

# for text in the plot
def plot_grid(states, values, offx, offy, th):
    for state, value in zip(states, values):
        plt.text(state[1]+offx, state[0]+offy, np.round(value, 1),
                 color='white' if value < th else 'k')


v_grid = make_grid(np.round(v, 1), [5, 5])
print(v_grid)
plt.imshow(v_grid)
plt.title("State_value function")
plt.colorbar()
plt.axis('off')
print(states)
plot_grid(states, v, -0.25, 0, 8)
plt.show()
