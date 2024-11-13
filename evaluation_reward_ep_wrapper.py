from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

class EvalRewardCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(EvalRewardCallback, self).__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for i in range(self.n_eval_episodes):
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=1,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
                print(f"Reward: {np.mean(episode_rewards)}, Episode length: {np.mean(episode_lengths)}")
        return True