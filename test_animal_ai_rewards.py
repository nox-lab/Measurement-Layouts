# Import the necessary environment wrappers.
from animalai.environment import AnimalAIEnvironment
import mlagents_envs # Provided by animalai
import mlagents_envs.envs.unity_gym_env
import stable_baselines3
import random

env_path = r"..\WINDOWS\AAI\Animal-AI.exe"
configuration_file = r"example.yaml"
port = 5005 + random.randint(
    0, 1000
)  # uses a random port to avoid problems with parallel runs
# Create the environment and wrap it...
env = AnimalAIEnvironment( # the environment object
    seed = 2023, # seed for the pseudo random generators
    file_name=env_path,
    arenas_configurations=configuration_file,
    play=False, # note that this is set to False for training
    base_port=port, # the port to use for communication between python and the Unity environment
    inference=True, # set to True if you want to watch the agent play
    useCamera=True, # set to False if you don't want to use the camera (no visual observations)
    resolution=64,
    useRayCasts=False, # set to True if you want to use raycasts
    no_graphics=False, # set to True if you don't want to use the graphics ('headless' mode)
    timescale=1.0, # the speed at which the simulation runs
)

environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port+1,
        arenas_configurations=configuration_file,
        play=True,
)

# Make it compatible with legacy Gym v0.21 API
env = mlagents_envs.envs.unity_gym_env.UnityToGymWrapper(
    env,
    uint8_visual=True,
    flatten_branched=True,  # Necessary if the agent doesn't support MultiDiscrete action space.
)

# Stable Baselines3 A2C model
# Will automatically use Shimmy to convert the legacy Gym v0.21 API to the Gymnasium API
model = stable_baselines3.A2C(
    "MlpPolicy",
    env,  # type: ignore
    device="cpu",
    verbose=1,
)
model.learn(total_timesteps=10_000)