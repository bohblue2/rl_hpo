
import ray
from ray import tune

from rl_hpo.agents.dqn import DQNAgent
from rl_hpo.params.dqn import DQNHyperParamsWithTune
from rl_hpo.trainer import Trainer


if __name__ == "__main__":
    ray.init()
    search_space = DQNHyperParamsWithTune(
        seed=42,
        discount_factor=tune.uniform(0.95, 0.99),
        learning_rate=tune.loguniform(1e-5, 1e-3),
        epsilon=1.0,
        epsilon_decay=tune.uniform(0.995, 0.99999),
        epsilon_min=0.001,
        batch_size=tune.choice([32, 64, 128]),
        train_start=tune.choice([500, 1000, 2000]),
        target_update_period=tune.choice([100, 200, 500]),
        hidden_size=tune.choice([64, 128, 256]),
        epochs=150,
        step_counter=100,
        target_score=200.0,
        env_name="LunarLander-v3",
        report_in_trial=False,
        use_wandb=True,
        save_model_interval=30
    )

    # Start hyperparameter optimization
    Trainer.optimize_hyperparameters(
        agent_class=DQNAgent,       # type: ignore
        search_space=search_space,  # type: ignore
        num_samples=20
    )