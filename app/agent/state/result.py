from typing import Any

from pydantic import BaseModel


class ResultState(BaseModel):
    # 训练计划
    workout_plan: Any | None = None

    # 饮食计划
    diet_plan: Any | None = None

    # 第一轮阶段性回答
    first_response: Any | None = None

    # 生成回答
    response: Any | None = None
    reflection_suggestions: Any | None = None
