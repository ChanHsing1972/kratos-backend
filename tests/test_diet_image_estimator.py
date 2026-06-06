import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.diet import MAX_FOOD_IMAGE_BYTES, estimate_from_image
from app.schemas.diet import FoodImageEstimateResult
from app.services.diet_image_estimator import (
    DietImageEstimatorError,
    estimate_food_from_image,
    normalize_food_estimate_payload,
)


class FakeUpload:
    def __init__(self, content: bytes, content_type: str = "image/jpeg") -> None:
        self.content = content
        self.content_type = content_type
        self.read_limit: int | None = None

    async def read(self, size: int = -1) -> bytes:
        self.read_limit = size
        return self.content if size < 0 else self.content[:size]


def test_food_image_estimate_normalizes_items_and_recomputes_total():
    result = normalize_food_estimate_payload(
        {
            "items": [
                {
                    "name": "米饭",
                    "estimated_weight_g": "180",
                    "estimated_kcal": 210,
                    "min_kcal": 170,
                    "max_kcal": 260,
                    "protein_g": 4,
                    "fat_g": 0.5,
                    "carbs_g": 46,
                    "confidence": 1.8,
                    "assumptions": "碗大小无法准确判断",
                    "source": "database",
                },
                {
                    "name": "鸡胸肉",
                    "estimated_kcal": 165.25,
                    "min_kcal": 140,
                    "max_kcal": 190,
                    "protein_g": 31.2,
                    "fat_g": 3.6,
                    "carbs_g": 0,
                    "confidence": 0.77,
                    "assumptions": ["烹饪油量不可见"],
                },
            ],
            "total": {"estimated_kcal": 9999},
            "need_user_confirmation": False,
        }
    )

    assert result.need_user_confirmation is True
    assert result.items[0].source == "ai_estimated"
    assert result.items[0].confidence == 1
    assert result.items[0].assumptions == ["碗大小无法准确判断"]
    assert result.total.estimated_kcal == 375.3
    assert result.total.min_kcal == 310
    assert result.total.max_kcal == 450
    assert result.total.protein_g == 35.2


def test_food_image_estimate_defaults_missing_fields():
    result = normalize_food_estimate_payload({"items": [{"name": "未知食物"}]})

    assert result.items[0].name == "未知食物"
    assert result.items[0].estimated_kcal == 0
    assert result.items[0].confidence == 0
    assert result.items[0].source == "ai_estimated"
    assert result.total.estimated_kcal == 0


def test_estimate_food_from_image_uses_responses_client_and_parses_json():
    class FakeResponses:
        def __init__(self) -> None:
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                output_text=(
                    '{"items":[{"name":"香蕉","estimated_kcal":90,'
                    '"min_kcal":70,"max_kcal":110,"protein_g":1.1,'
                    '"fat_g":0.3,"carbs_g":23,"confidence":0.8}]}'
                )
            )

    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)

    result = estimate_food_from_image(
        image_bytes=b"fake-image",
        mime_type="image/png",
        client=client,
    )

    assert result.items[0].name == "香蕉"
    assert result.total.estimated_kcal == 90
    assert responses.request["model"]
    image_part = responses.request["input"][0]["content"][1]
    assert image_part["image_url"].startswith("data:image/png;base64,")


def test_estimate_food_from_image_falls_back_to_agent_llm(monkeypatch):
    class FakeAgentLLM:
        def __init__(self) -> None:
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(
                content=(
                    '{"items":[{"name":"苹果","estimated_kcal":80,'
                    '"min_kcal":60,"max_kcal":100,"protein_g":0.3,'
                    '"fat_g":0.2,"carbs_g":21,"confidence":0.82}]}'
                )
            )

    fake_llm = FakeAgentLLM()
    monkeypatch.setattr(
        "app.services.diet_image_estimator.settings.OPENAI_API_KEY",
        None,
    )
    monkeypatch.setattr("app.services.diet_image_estimator.get_food_agent_llm", lambda: fake_llm)

    result = estimate_food_from_image(
        image_bytes=b"fake-image",
        mime_type="image/jpeg",
    )

    assert result.items[0].name == "苹果"
    assert result.total.estimated_kcal == 80
    content = fake_llm.messages[0].content
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_estimate_food_from_image_rejects_invalid_json():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(output_text="not json")

    client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(DietImageEstimatorError):
        estimate_food_from_image(
            image_bytes=b"fake-image",
            mime_type="image/jpeg",
            client=client,
        )


def test_estimate_food_from_image_wraps_llm_call_errors():
    class FakeResponses:
        def create(self, **kwargs):
            raise RuntimeError("secret stack with key")

    client = SimpleNamespace(responses=FakeResponses())

    with pytest.raises(DietImageEstimatorError):
        estimate_food_from_image(
            image_bytes=b"fake-image",
            mime_type="image/jpeg",
            client=client,
        )


def test_diet_endpoint_rejects_missing_image():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(estimate_from_image(image=None, current_user=SimpleNamespace(id=1)))

    assert exc_info.value.status_code == 400


def test_diet_endpoint_rejects_unsupported_type():
    upload = FakeUpload(b"hello", content_type="text/plain")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(estimate_from_image(image=upload, current_user=SimpleNamespace(id=1)))

    assert exc_info.value.status_code == 400
    assert "JPG" in exc_info.value.detail


def test_diet_endpoint_rejects_empty_image():
    upload = FakeUpload(b"", content_type="image/webp")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(estimate_from_image(image=upload, current_user=SimpleNamespace(id=1)))

    assert exc_info.value.status_code == 400
    assert "不能为空" in exc_info.value.detail


def test_diet_endpoint_rejects_too_large_image():
    upload = FakeUpload(b"x" * (MAX_FOOD_IMAGE_BYTES + 1), content_type="image/png")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(estimate_from_image(image=upload, current_user=SimpleNamespace(id=1)))

    assert exc_info.value.status_code == 400
    assert upload.read_limit == MAX_FOOD_IMAGE_BYTES + 1


def test_diet_endpoint_returns_wrapped_success(monkeypatch):
    def fake_estimate_food_from_image(*, image_bytes: bytes, mime_type: str):
        assert image_bytes == b"image"
        assert mime_type == "image/jpeg"
        return FoodImageEstimateResult(
            items=[],
            total={},
            warning="该结果为 AI 估算，可能受到拍摄角度、食物遮挡、油量、酱料和份量判断误差影响。",
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.diet.estimate_food_from_image",
        fake_estimate_food_from_image,
    )

    response = asyncio.run(
        estimate_from_image(
            image=FakeUpload(b"image", content_type="image/jpeg"),
            current_user=SimpleNamespace(id=1),
        )
    )

    assert response.success is True
    assert response.data.need_user_confirmation is True


def test_diet_endpoint_maps_estimator_errors_to_502(monkeypatch):
    def fake_estimate_food_from_image(*, image_bytes: bytes, mime_type: str):
        raise DietImageEstimatorError("OPENAI_API_KEY=secret")

    monkeypatch.setattr(
        "app.api.v1.endpoints.diet.estimate_food_from_image",
        fake_estimate_food_from_image,
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            estimate_from_image(
                image=FakeUpload(b"image", content_type="image/jpeg"),
                current_user=SimpleNamespace(id=1),
            )
        )

    assert exc_info.value.status_code == 502
    assert "secret" not in exc_info.value.detail
