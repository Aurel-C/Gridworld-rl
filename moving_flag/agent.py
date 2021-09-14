import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import seaborn as sns
import matplotlib.pyplot as plt

SIDE_L = 9


class DenseNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x[0]


class Agent:
    def __init__(self, alpha=0.01, gamma=0.8, lambda_=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tau = 1e-2
        self.key = jax.random.PRNGKey(51)
        self.key, subkey = jax.random.split(self.key)
        self.w = DenseNet().init(subkey, jnp.zeros((10,)))
        self.z = jax.tree_multimap(lambda w: w * 0.0, self.w)
        self.reward = 0

    def update_params(self, episode):
        self.tau = 1.0 / (1.0 + episode / 3.0)
        self.alpha *= 0.99

    def update_weights(self, reward):
        self.w, self.z = update(
            self.w,
            self.z,
            self.v,
            self.vp,
            self.grad,
            reward,
            self.alpha,
            self.gamma,
            self.lambda_,
        )

    def get_action(self, state, end):
        a, self.v, self.vp, self.grad, self.key = get_action(
            self.w, jnp.array(state), jnp.array(end), self.tau, self.key
        )
        return a

    def plot_grid(self, end):
        a = jnp.arange(0, SIDE_L + 1)
        x, y = jnp.meshgrid(a, a)
        s = jnp.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
        xs = jax.vmap(jax.partial(poly_input, end=end))(s)
        get_values = jax.partial(DenseNet().apply, self.w)
        values = jax.jit(jax.vmap(get_values))(xs)
        plt.clf()
        plt.imshow(
            values.reshape((SIDE_L + 1, SIDE_L + 1)), cmap="magma", origin="lower"
        )
        plt.colorbar()
        plt.show()

    def load_weights(self, name):
        with open(name, "rb") as f:
            self.w = serialization.from_bytes(self.w, f.read())


@jax.jit
def get_action(w, state, end, tau, key):
    x = poly_input(state, end)
    value, grad = jax.value_and_grad(DenseNet().apply)(w, x)
    branches = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
    #                       Up     Down     Left    Right
    s2 = state + branches
    s2 = jnp.clip(s2, 0, SIDE_L)

    next_x = jax.vmap(jax.partial(poly_input, end=end))(s2)
    get_values = jax.partial(DenseNet().apply, w)
    next_values = jax.vmap(get_values)(next_x)
    expected = jnp.dot(next_values, nn.softmax(next_values / tau))
    a, key = softmax_action(next_values, tau, key)
    return a, value, expected, grad, key


@jax.jit
def update(w, z, v, next_v, grad, reward, alpha, gamma, lambda_):
    # Updates
    z = jax.tree_multimap(lambda z, grad: z * lambda_ * gamma + grad, z, grad,)
    w = jax.tree_multimap(
        lambda w, z: w + alpha * z * (reward + gamma * next_v - v), w, z,
    )
    return w, z


def poly_input(state, end):
    state = (state - SIDE_L / 2) / (SIDE_L / 2)
    end = (end - SIDE_L / 2) / (SIDE_L / 2)
    return jnp.concatenate(
        [state, end, jnp.square(state), jnp.square(end), state * end]
    )


def softmax_action(values, tau, key):
    policy = nn.softmax(values / tau)
    key, subkey = jax.random.split(key)
    a = jax.random.choice(subkey, jnp.arange(4), p=policy)
    return a, key

def main():
    '''Train the agent without unity 
        Faster but much slower than the full jax implementation
    '''
    agent = Agent()
    key = jax.random.PRNGKey(52)
    episodes = []

    def move(state, action):
        branches = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]])
        s2 = state + branches[action]
        s2 = jnp.clip(s2, 0, SIDE_L)
        return s2

    for i in range(100):
        steps = 0
        state = jnp.array([4, 4])
        key, subkey = jax.random.split(key)
        end = jax.random.randint(subkey, (2,), 0, SIDE_L + 1)
        agent.update_params(i)
        cond = True
        while cond:
            steps += 1
            action = agent.get_action(state, end)
            state = move(state, action)
            if jnp.all(state == end):
                reward = 1.0
                cond = False
            else:
                reward = -0.01
            agent.update_weights(reward)
        print(steps)
        episodes.append(steps)

if __name__ == "__main__":
    main()

