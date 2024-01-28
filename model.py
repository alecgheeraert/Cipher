from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, DQN, A2C
from controller import Controller

class Model:
    def __init__(self, 
        env, 
        time_steps=30000, 
        model_type="PPO",
        verbose=1
        ):
        """
        Parameters
        ----------
        env : Controller
            The environment to train the model on.
        time_steps : int, optional
            The number of time steps to train the model for. The default is 30000.
        model_type : str, optional
            The type of model to train. The default is "PPO".
        verbose : int, optional
            The verbosity level. The default is 1.
        """
        self.env = DummyVecEnv([lambda: env])
        self.model_type = model_type
        self.time_steps = time_steps
        self.verbose = verbose
        self._models()

    def _models(self):  
        if self.model_type == "PPO":
            self.model = PPO("MultiInputPolicy", self.env, verbose=self.verbose)
        elif self.model_type == "DQN":
            self.model = DQN("MultiInputPolicy", self.env, verbose=self.verbose)
        elif self.model_type == "A2C":
            self.model = A2C("MultiInputPolicy", self.env, verbose=self.verbose)
        else:
            raise ValueError("Invalid model type.")

    def learn(self):
        self.model.learn(total_timesteps=self.time_steps)
        self.model.save("research")

    def evaluate(self):
        loaded_model = PPO.load("research")
        mean_reward, std_reward = evaluate_policy(loaded_model, self.env, n_eval_episodes=10)
        print(mean_reward, std_reward)

    def replay(self):
        loaded_model = PPO.load("research")
        obs = self.env.reset()
        for _ in range(1000):
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()


model = Model(
    Controller(20.0, 0.1, render_mode="human"),
)
model.learn()
model.evaluate()
model.replay()