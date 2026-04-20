from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserBase(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    gender: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = Field(default=None, max_length=100)
    dietary_habits: str | None = None
    fitness_status: str | None = None


class UserCreate(UserBase):
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class UserUpdate(BaseModel):
    gender: str | None = Field(default=None, max_length=20)
    age: int | None = Field(default=None, ge=0, le=120)
    location: str | None = Field(default=None, max_length=100)
    dietary_habits: str | None = None
    fitness_status: str | None = None

    model_config = ConfigDict(extra="forbid")

    def to_update_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str | None = None
