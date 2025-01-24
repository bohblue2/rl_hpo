
import gymnasium as gym
from typing import Any, Dict, List, Type, TypeVar
import dill # type: ignore 
import numpy as np
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import wandb
from rl_hpo.agents.dqn import DQNAgent
from rl_hpo.baes import HyperParams
from rl_hpo.common import init_seed
from rl_hpo.constants import STATE_INDEX


T = TypeVar("T", bound=DQNAgent)

class Trainer:
    def __init__(self, agent_class: Type[T], config: Dict[str, Any]):
        self.agent_class = agent_class
        self.config = config
        init_seed(self.config.get('seed', 0))
        self.env = gym.make(self.config.get('env_name', 'LunarLander-v3'))
        self.state_size = self.env.observation_space.shape[STATE_INDEX] # type: ignore
        self.action_size = self.env.action_space.n                      # type: ignore
        self.agent: Type[T] = self.agent_class(
            self.state_size, self.action_size, self.config              # type: ignore
        )
        self.scores: List[float] = []
        self.episodes: List[float] = []

    def train(self):
        wandb_runner = None
        if self.config.get('use_wandb', False):
            wandb_runner = wandb.init(
                project=self.config.get('project_name', "rl_hpo"), 
                config=self.config, 
                reinit=True
            )
            wandb_runner.define_metric("epoch_step")
            wandb_runner.define_metric("score_epoch", step_metric="epoch_step")

        EPOCHS = self.config.get('epochs', None)
        TARGET_SCORE = self.config.get('target_score', None)

        for epoch in range(EPOCHS):
            done = False
            truncated = False
            score = 0

            state, _ = self.env.reset()
            while not (done or truncated):
                action = self.agent.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.train()
                score += reward
                state = next_state

                if self.agent.step_counter % self.config.get('step_counter', 0) == 0:
                    if self.config.get("use_wandb", False):
                        wandb_runner.log({
                            "score": score,
                            "epsilon": self.agent.epsilon,
                            "loss": self.agent.loss_history[-1] if self.agent.loss_history else 0,
                            "step_counter": self.agent.step_counter
                        })

            self.agent.update()
            self.scores.append(score)
            self.episodes.append(epoch)
            avg_score = np.mean(self.scores[-min(30, len(self.scores)):])

            print(
                f"Episode: {epoch}, "
                f"Score: {score:.2f}, "
                f"Average Score: {avg_score:.2f}, "
                f"Epsilon: {self.agent.epsilon:.4f}"
            )
            if wandb_runner:
                wandb_runner.log({
                    "epoch_step": epoch + 1, 
                    "avg_score": avg_score,
                    "score_epoch": score
                })
            if epoch % self.config.get('save_model_interval', 0) == 0:
                if wandb_runner:
                    model_state_dict = self.agent.policy_net.state_dict()
                    dill.dump(model_state_dict, open(f"model-{epoch}.pkl", "wb"))
                    artifact = wandb.Artifact(
                        f"{wandb_runner.id}-{epoch}", 
                        type="model",
                        description="Model Checkpoint",
                        metadata={
                            "avg_score": avg_score,
                            "epoch": epoch
                        }                        
                    )
                    artifact.add_file(f"model-{epoch}.pkl")
                    wandb_runner.log_artifact(artifact)

            if avg_score >= TARGET_SCORE:
                wandb_runner.log({"solved_episode": epoch + 1})
                break
            if self.config.get('report_in_trial', False):
               train.report({'avg_score': avg_score}) 


        train.report({'avg_score': avg_score})
        self.env.close()
    
    @classmethod
    def optimize_hyperparameters(
        cls,
        agent_class: Type[T],
        search_space: HyperParams,
        num_samples: int = 10,
        max_concurrent_trials=1
    ):
        def train_wrapper(config_dict):
            trainer = cls(agent_class, config_dict)
            trainer.train()

        # HyperOpt and ASHA Scheduler
        algo = HyperOptSearch(metric="avg_score", mode="max")
        scheduler = ASHAScheduler(metric="avg_score", mode="max")

        # Run hyperparameter optimization
        tune.run(
            train_wrapper,
            config=search_space.model_dump(),
            scheduler=scheduler,
            search_alg=algo,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials
        )
