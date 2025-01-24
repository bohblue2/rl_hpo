
from typing import Optional
from pydantic import BaseModel, Field
import wandb

class WandbRunData(BaseModel):
    name: str
    run_id: str
    project: str
    summary: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    artifact: Optional[wandb.Artifact] = None
    artifact_filepath: str = "" 

    class Config:
        arbitrary_types_allowed = True
    
    def get_artifact(
        self, 
        epoch: int, 
        type: str = 'model',
        download: bool = False,
        root: str = '.artifacts'
    ):
        try:
            artifact = wandb.Api().artifact(
                f'{self.project}/{self.run_id}-{epoch}:latest',
                type=type
            )
        except Exception as e:
            print(f"Error downloading artifact: {e}")
            return None
        else:
            if download:
                self.artifact_filepath = artifact.file(
                    root=root
                )
            self.artifact = artifact
            return artifact