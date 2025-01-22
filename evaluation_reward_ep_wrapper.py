from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pandas as pd
from demands import Demands

class EvalRewardCallback(EvalCallback):
    def __init__(self, results_file, instances : list[Demands] = None, aai_env = None, config_file: str = None, *args, **kwargs):
        super(EvalRewardCallback, self).__init__(*args, **kwargs)
        self.instances = instances
        self.results_file = results_file
        self.aai_env = aai_env
        self.config = config_file
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            for i in range(self.n_eval_episodes):
                instance_demands = self.instances[i]
                print(instance_demands)
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
                data = {
                    'Xpos': instance_demands.Xpos,
                    'reward_behind': instance_demands.reward_behind,
                    'reward_distance': instance_demands.reward_distance,
                    'reward_size': instance_demands.reward_size,
                    'reward': np.mean(episode_rewards)
                }
                df = pd.DataFrame([data])
                df.to_csv(self.results_file, mode='a', header=not pd.io.common.file_exists(self.results_file), index=False)
                
            self.aai_env.reset(arenas_configurations=self.config)
        return True