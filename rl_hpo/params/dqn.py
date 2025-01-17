from typing import Any
from pydantic import BaseModel, Field
from ray import tune


class DQNHyperParamsWithTune(BaseModel):
    seed: int = Field(42, title="Seed for random number generator")
    env_name: str = Field("LunarLander-v3", title="Name of the environment")
    discount_factor: Any = Field(
        tune.uniform(0.95, 0.99),
        description="Discount factor for future rewards"
    )
    learning_rate: Any = Field(
        tune.loguniform(1e-5, 1e-3),
        description="Learning rate for the optimizer"
    )
    epsilon: Any = Field(
        1.0,
        description="Initial exploration rate"
    )
    epsilon_decay: Any = Field(
        tune.uniform(0.995, 0.99999),
        description="Decay rate for the exploration rate"
    )
    epsilon_min: Any = Field(
        0.001,
        description="Minimum exploration rate"
    )
    batch_size: Any = Field(
        tune.choice([32, 64, 128]),
        description="Batch size for training"
    )
    train_start: Any = Field(
        tune.choice([500, 1000, 2000]),
        description="Number of experiences before training starts"
    )
    target_update_period: Any = Field(
        tune.choice([100, 200, 500]),
        description="Steps before updating the target network"
    )
    hidden_size: Any = Field(
        tune.choice([64, 128, 256]),
        description="Number of neurons in hidden layers"
    )
    epochs: Any = Field(
        1000,
        description="Total training epochs"
    )
    step_counter: int = Field(
        100,
        description="Step interval for logging metrics like score, epsilon and other parameters"
    )
    target_score: Any = Field(
        200.0,
        description="Score at which the environment is considered solved"
    )
    report_in_trial: bool = Field(
        False,
        description="Report average score in each trial"
    )
    use_wandb: bool = Field(
        False,
        description="Use Weights and Biases for logging"
    )
    save_model_interval: int = Field(
        30,
        description="Interval for saving the model in wandb"
    )
    
    
    class Config:
        arbitrary_types_allowed = True