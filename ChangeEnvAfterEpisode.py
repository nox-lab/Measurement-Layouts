from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import random
from generating_configs import gen_config_from_demands
from demands import Demands
class EndofEpisodeReward(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, aai_env, instances : list[Demands] = None,  config_str : str = None, verbose=0):
        super(EndofEpisodeReward, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.config = config_str
        self.aai_env = aai_env
        self.num_episodes = 0
        self.wins = 0 
        self.instances = instances
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if not np.sum(self.locals["dones"]).item():
            return True
        self.num_episodes += np.sum(self.locals["dones"]).item()
        print("Episode finished")
        final_score = self.locals["infos"][0]["episode"]["r"]
        if final_score < -0.9 and self.num_episodes < 100 and not self.instances:
            return True
        print(final_score)
        print(self.aai_env)
        
        instance_point = self.num_episodes % len(self.instances)
        if not self.instances:
            self.num_episodes = 0
            new_xpos = random.choice([-1, 0, 1])
            new_behind = random.choice([0, 0.5, 1])
            new_size = np.random.uniform(0, 1.9)
            new_distance = np.random.uniform(new_size+0.5, 5.3)
        else:
            new_xpos = self.instances[instance_point].Xpos
            new_behind = self.instances[instance_point].reward_behind
            new_size = self.instances[instance_point].reward_size
            new_distance = self.instances[instance_point].reward_distance
        if new_behind != 0:
            time_limit = 100
        else:
            time_limit = 20
        gen_config_from_demands(new_size, new_distance, new_behind, new_xpos, time_limit, self.config)
        self.aai_env.reset(arenas_configurations=self.config)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass