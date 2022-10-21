import numpy as np

def train(ph=0.4, theta=0.0001, gamma=0.9):
    # Action space is in range(0,100).
    # Set an intial value for state.
    value_estimates = np.random.rand(100)
    value_estimates[0] = 0
    value_estimates[99] = 1
    pi = np.zeros(100)
    counter = 0
    # Value iteration
    while True:
        delta = 0
        for s in range(1, 100):
            old_value_estimates = value_estimates
            action_value_estimates = np.zeros(s+1)
            # update the estimate of all actions that we can take in state s
            for a in range(1, 100-s):
                if a + s <= 100:
                    # the expected reward of every state transition is 0, except for transitioning to state 100
                    # The reward for state 100 is +1
                    action_value_estimates[s] += ph*(0+gamma*value_estimates[a+s])
                    action_value_estimates[s] += (1-ph)*(0+gamma*value_estimates[a-s])
            arg_optimal_action = np.argmax(action_value_estimates)
            pi[s] = arg_optimal_action
            value_estimates[s] = action_value_estimates[arg_optimal_action]
            delta = max(abs(np.amax(old_value_estimates-value_estimates)), delta)
        counter += 1
        if counter % 1000 == 0:
            print(f"Train loop: {counter}")
            print(f"Delta: {delta}")
        # Convergence criteria
        if delta < theta:
            break
        return value_estimates[1:100], pi[1:100]


if __name__ == "__main__":
    train()
