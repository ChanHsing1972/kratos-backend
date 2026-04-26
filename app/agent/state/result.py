from pydantic import BaseModel
from typing import Any


class ResultState(BaseModel):
    # 训练计划
    workout_plan: Any | None = None

    # 饮食计划
    diet_plan: Any | None = None

    # 生成回答
    response: Any | None = None
    reflection_suggestions: Any | None = None
