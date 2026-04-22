from typing import Any

from pydantic.v1 import BaseModel


class LongTermMemory(BaseModel):
    # 用户基本信息
    name: str | None = None
    gender: str | None = None
    job: str | None = None


class MidTermMemory(BaseModel):
    # 后期这四个应该会打包成一个类
    train_id: int | None = None
    # 计划
    plans: list[dict[str, Any]] = []
    # 完成度
    completions: list[int] = []
    # 反馈
    feedbacks: list[str] = []


class MemoryState(BaseModel):
    long_term_memory: LongTermMemory = LongTermMemory()
    mid_term_memory: MidTermMemory = MidTermMemory()
