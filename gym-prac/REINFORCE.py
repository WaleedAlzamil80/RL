from helpful import *

class REINFORCE:
    """
    REINFORCE algorithm implementation.
    First-visit MC for estimating the state-value function Given policy.
    """
    def __init__(self, env_info, reward_norm = False):
        self.logprobs = []
        self.rewards = []
        self.losses = []
        self.gamma = 0.99
        self.lr = 1e-4
        self.eps = 1e-6
        self.continous = env_info["action_space_continuous"]

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.policy = PolicyNetwork(env_info).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        if self.continous:
            mu, log_std = self.policy(state)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()
            self.logprobs.append(log_prob)
            return torch.tanh(action).cpu().detach().numpy()

        action_probs = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.logprobs.append(dist.log_prob(action))
        return action.item()

    def update(self):
        running_gs = 0
        # Policy Evaluation
        Gs = []
        for R in self.rewards[::-1]:
            running_gs = running_gs * self.gamma + R
            Gs.insert(0, running_gs)

        Gs = torch.tensor(Gs, dtype=torch.float32).to(self.device)

        loss = 0
        for log_prob, G in zip(self.logprobs, Gs):
            loss += -log_prob * G

        self.losses.append(loss.item())
        # Policy Improvement
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset the rewards and the likilihood
        self.logprobs = []
        self.rewards = []

    def combined_episode_videos(self):
        video_names = np.asarray(glob('gym-prac/videos'+'/*.mp4'))
        if video_names.shape[0]==0: return
        cap = cv2.VideoCapture(video_names[0])
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('gym-prac/combined_{}.mp4'.format(self.params.env_name),cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width,frame_height))
        for idx,video in enumerate(video_names):
            episode_idx = idx*self.params.log_episode_interval
            cap = cv2.VideoCapture(video)
            counter = 0
            while(True):
                font_size = self.params.font_size*(frame_width/600)
                if font_size<0.5:
                    font_size = 0.5
                margin = int(self.params.font_margin/600*frame_width)
                # Capture frames in the video
                ret, frame = cap.read()

                if not ret:
                    break
                font = cv2.FONT_HERSHEY_SIMPLEX

                cv2.putText(frame,'Episode: {}'.format(episode_idx+1),(margin, margin),font, font_size,self.params.font_color,2, cv2.LINE_4)
                cv2.putText(frame,'Reward: {:.2f}'.format(self.episode_rewards[episode_idx]), (margin, frame_height-margin), font, font_size,self.params.font_color, 2,cv2.LINE_4)
                out.write(frame)
                counter += 1

            cap.release()
        out.release()

    def save_nets(self,pth_name):
        torch.save(self.policy.state_dict(), f"{pth_name}_policy_net.pth")