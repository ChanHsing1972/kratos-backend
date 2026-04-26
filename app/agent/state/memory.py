from typing import Any

from pydantic import BaseModel, Field


class LongTermMemory(BaseModel):
    # 用户基本信息
    name: str | None = None
    gender: str | None = None
    job: str | None = None


class MidTermMemory(BaseModel):
    # 后期这四个应该会打包成一个类
    train_id: int | None = None
    # 计划
    plans: list[dict[str, Any]] = Field(default_factory=list)
    # 完成度
    completions: list[int] = Field(default_factory=list)
    # 反馈
    feedbacks: list[str] = Field(default_factory=list)


class MemoryState(BaseModel):
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    mid_term_memory: MidTermMemory = Field(default_factory=MidTermMemory)
