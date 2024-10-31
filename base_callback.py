from stable_baselines3.common.callbacks import BaseCallback
class StopOnTruncCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self):
        return self._is_trunc()

    def _is_trunc(self):
        return self.training_env.get_attr("truncated")[0]