from pydantic import BaseModel, Field
from typing import Dict, Any

class ModelNameConfig(BaseModel):
    """Model Configuration"""

    model_name: str = Field("randomforest", description="Name of the model to be used in the pipeline.")
    fine_tuning: bool = False
    