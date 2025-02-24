from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import pandas as pd
from demands import Demands
from animalai.environment import AnimalAIEnvironment
from animal_ai_reset_wrapper import AnimalAIReset

class EvalRewardCallback(EvalCallback):
    def __init__(self, results_file, instances : list[Demands] = None, aai_env: AnimalAIReset = None, config_file: str = None, best_model_path : str = None, num_evals = None, *args, **kwargs):
        super(EvalRewardCallback, self).__init__(*args, **kwargs)
        self.instances = instances
        self.results_file = results_file
        self.aai_env = aai_env
        self.config = config_file
        self.best_model_path = best_model_path
        self.best_reward = -np.inf
        self.number_of_evals = 0
        self.number_of_evals_max = np.inf if num_evals is None else num_evals
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.aai_env.reset_arenas() # Reset the environment back to its initial state.
            self.total_reward = 0
            self.number_of_evals += 1
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
                self.total_reward += np.mean(episode_rewards)
                data = {
                    'Xpos': instance_demands.Xpos,
                    'reward_behind': instance_demands.reward_behind,
                    'reward_distance': instance_demands.reward_distance,
                    'reward_size': instance_demands.reward_size,
                    'reward': np.mean(episode_rewards)
                }
                df = pd.DataFrame([data])
                df.to_csv(self.results_file, mode='a', header=not pd.io.common.file_exists(self.results_file), index=False)
            if self.total_reward > self.best_reward and self.best_model_path:
                print(f"New best reward: {self.total_reward}, saving to {self.best_model_path}")
                self.best_reward = self.total_reward
            if self.best_model_path:    
                self.model.save(self.best_model_path+ f"_{self.n_calls}.zip")
                print("model saved!")
            self.aai_env.reset(arenas_configurations=self.config)
        if self.number_of_evals >= self.number_of_evals_max:
            return False
        return True