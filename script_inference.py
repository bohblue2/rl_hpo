from typing import Any, Dict, List
import pandas as pd
import wandb
import dill # type: ignore
import gymnasium as gym
from rl_hpo.agents.dqn import DQNAgent
from rl_hpo.enums import WandbConfigKeys, WandbMetrics
from rl_hpo.models.wandb import WandbRunData

def load_best_model_and_run_inference(project_name: str, epoch: int = 30):
    api = wandb.Api()
    runs = api.runs(f'ryanbae/{project_name}', filters={"state": "finished"})
    run_data = {}
    for run in runs:
        summary = {k: v for k, v in run.summary.items() if not k.startswith("_")}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        run_data_instance = WandbRunData(
            name=run.name,
            run_id=run.id,
            project=run.project,
            summary=summary,
            config=config
        )
        run_data[run.id] = run_data_instance

    data_list: List[Dict[str, Any]] = []
    for run_id, run_data_instance in run_data.items():
        data_list.append({
            "run_id": run_data_instance.run_id,
            "run_name": run_data_instance.name,
            "run_data_instance": run_data_instance,
            WandbConfigKeys.DISCOUNT_FACTOR.value: run_data_instance.config.get(WandbConfigKeys.DISCOUNT_FACTOR.value),
            WandbConfigKeys.LEARNING_RATE.value: run_data_instance.config.get(WandbConfigKeys.LEARNING_RATE.value),
            WandbConfigKeys.EPSILON.value: run_data_instance.config.get(WandbConfigKeys.EPSILON.value),
            WandbConfigKeys.EPSILON_DECAY.value: run_data_instance.config.get(WandbConfigKeys.EPSILON_DECAY.value),
            WandbConfigKeys.EPSILON_MIN.value: run_data_instance.config.get(WandbConfigKeys.EPSILON_MIN.value),
            WandbConfigKeys.BATCH_SIZE.value: run_data_instance.config.get(WandbConfigKeys.BATCH_SIZE.value),
            WandbConfigKeys.TRAIN_START.value: run_data_instance.config.get(WandbConfigKeys.TRAIN_START.value),
            WandbConfigKeys.TARGET_UPDATE_PERIOD.value: run_data_instance.config.get(WandbConfigKeys.TARGET_UPDATE_PERIOD.value),
            WandbConfigKeys.HIDDEN_SIZE.value: run_data_instance.config.get(WandbConfigKeys.HIDDEN_SIZE.value),
            WandbMetrics.AVG_SCORE.value: run_data_instance.summary.get(WandbMetrics.AVG_SCORE.value), 
            WandbMetrics.EPOCH_STEP.value: run_data_instance.summary.get(WandbMetrics.EPOCH_STEP.value),
            WandbMetrics.EPSILON.value: run_data_instance.summary.get(WandbMetrics.EPSILON.value),
            WandbMetrics.LOSS.value: run_data_instance.summary.get(WandbMetrics.LOSS.value),
            WandbMetrics.SCORE.value: run_data_instance.summary.get(WandbMetrics.SCORE.value),
        })

    df = pd.DataFrame(data_list)
    df_sorted = df.sort_values(WandbMetrics.AVG_SCORE.value, ascending=False)
    df_sorted.dropna(inplace=True)

    if not df_sorted.empty:
        best_run = df_sorted.iloc[0]
        run_data_instance = best_run["run_data_instance"]
        run_data_instance.get_artifact(epoch=epoch)
        model_path = run_data_instance.artifact_filepath
        model_state_dict = dill.load(open(model_path, 'rb'))
        
        env = gym.make(run_data_instance.config.get('env_name', 'LunarLander-v3'))
        state_size = env.observation_space.shape[0] # type: ignore  
        action_size = env.action_space.n            # type: ignore
        agent = DQNAgent(state_size, action_size, run_data_instance.config)
        agent.policy_net.load_state_dict(model_state_dict)
        agent.policy_net.eval()

        # Run inference
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action = agent.get_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward # type: ignore
            state = next_state

        print(f"Total Reward: {total_reward}")
        env.close()
    else:
        print("No completed runs found.")

load_best_model_and_run_inference("rl_hpo")