from helpful import *
from gymnasium.wrappers import RecordVideo

class ActorCritic:
    """
    Actor-Critic algorithm implementation.
    """
    def __init__(self, env_name):
        self.env_name = env_name
        self.rewards = []
        self.gamma = 0.99
        self.lr = 1e-4
        self.eps = 1e-6
        self.episodes = 2500
        env_info = query_env(self.env_name)
        self.continous = env_info['action_space_continuous']

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.actor = PolicyNetwork(env_info).to(self.device)
        self.critic = ValueNetwork(env_info).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)


    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if self.continous:
            mu, log_std = self.actor(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            sample = dist.sample()
            log_prob = dist.log_prob(sample).sum()
            action = torch.tanh(sample).cpu().detach().numpy()
            return action, log_prob

        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        action =  sample.item()
        return action, log_prob


    def Train(self):
        """
        TD(0) algorithm implementation.
        """
        env = gym.make(self.env_name, render_mode = 'rgb_array')
        env.metadata['render_fps'] = 30
        value_criterion = torch.nn.MSELoss()

        for episode in range(self.episodes):
            state, _ = env.reset()
            done = False
            while not done:
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                action, log_prob = self.sample_action(state)

                if self.continous:
                    next_state, reward, done, truncated, info = env.step(action)
                else:
                    next_state, reward, done, truncated, info = env.step(int(action))
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                value = self.critic(state)
                target = reward + (1 - done) * self.gamma * self.critic(next_state)
                vloss = value_criterion(value, target.detach())

                advantage = (target - value).detach()
                aloss = -log_prob * advantage

                self.critic_optimizer.zero_grad()
                vloss.backward()
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                aloss.backward()
                self.actor_optimizer.step()

                state = next_state
                done = done or truncated
                if (episode + 1) % 100 == 0:
                    self.rewards.append(reward)

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1} : " , "Reward : ", int(np.sum(np.array(self.rewards))))
                self.rewards = []

        self.rewards = []

    def save_nets(self,pth_name):
        torch.save(self.actor.state_dict(), f"{pth_name}_policy_net.pth")
        torch.save(self.critic.state_dict(), f"{pth_name}_value_net.pth")

agent = ActorCritic("CartPole-v1")
agent.Train()