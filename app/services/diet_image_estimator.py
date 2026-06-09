from __future__ import annotations

import base64
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from io import BytesIO
from typing import Any

from app.agent.json_utils import LLMJsonParseError, parse_json_object
from app.core.config import settings
from app.schemas.diet import FoodEstimateItem, FoodEstimateTotal, FoodImageEstimateResult

FOOD_ESTIMATE_WARNING = "该结果为 AI 估算，可能受到拍摄角度、食物遮挡、油量、酱料和份量判断误差影响。"

FOOD_IMAGE_ESTIMATE_PROMPT = f"""
你是一个饮食热量估算助手。请根据图片识别食物，并估算每种食物的份量和营养成分。
你必须只输出 JSON，不要输出 Markdown、解释文字或代码块。
如果图片中有多个食物，请分别列出。
热量和营养成分只能估算，不要假装精确。
请给出 estimated_kcal、min_kcal、max_kcal，体现误差范围。
如果无法判断重量、食材、油量、糖量、酱料或烹饪方式，请降低 confidence，并在 assumptions 中说明。
confidence 范围是 0 到 1。
所有数值字段使用数字，不要使用字符串。
如果无法识别食物，items 返回空数组，并设置 need_user_confirmation 为 true。

输出 JSON 格式必须符合：

{{
  "items": [
    {{
      "name": "食物名称",
      "estimated_weight_g": 0,
      "estimated_kcal": 0,
      "min_kcal": 0,
      "max_kcal": 0,
      "protein_g": 0,
      "fat_g": 0,
      "carbs_g": 0,
      "confidence": 0.0,
      "assumptions": [],
      "source": "ai_estimated"
    }}
  ],
  "total": {{
    "estimated_kcal": 0,
    "min_kcal": 0,
    "max_kcal": 0,
    "protein_g": 0,
    "fat_g": 0,
    "carbs_g": 0
  }},
  "need_user_confirmation": true,
  "warning": "{FOOD_ESTIMATE_WARNING}"
}}
""".strip()


class DietImageEstimatorError(RuntimeError):
    """Raised when the food image estimator cannot call or parse the LLM."""


def estimate_food_from_image(
    *,
    image_bytes: bytes,
    mime_type: str,
    client: Any | None = None,
) -> FoodImageEstimateResult:
    """Estimate food and nutrition from an image without persisting it."""

    model_image_bytes, model_mime_type = _prepare_image_for_model(
        image_bytes=image_bytes,
        mime_type=mime_type,
    )
    image_base64 = base64.b64encode(model_image_bytes).decode("ascii")
    data_url = f"data:{model_mime_type};base64,{image_base64}"

    if client is not None:
        content = _estimate_with_vision_client(client, data_url=data_url)
    elif settings.FOOD_VISION_API_KEY or settings.OPENAI_API_KEY:
        content = _estimate_with_vision_client(
            get_food_vision_client(),
            data_url=data_url,
        )
    else:
        content = _estimate_with_agent_llm(data_url=data_url)

    try:
        payload = parse_json_object(content)
    except LLMJsonParseError as exc:
        raise DietImageEstimatorError("food vision model returned invalid JSON") from exc

    return normalize_food_estimate_payload(payload)


@lru_cache(maxsize=1)
def get_food_vision_client() -> Any:
    api_key = settings.FOOD_VISION_EFFECTIVE_API_KEY
    if not api_key:
        raise DietImageEstimatorError("food vision API key is not configured")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise DietImageEstimatorError("openai package is not installed") from exc

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "default_headers": settings.OPENAI_COMPAT_DEFAULT_HEADERS,
        "max_retries": settings.FOOD_VISION_MAX_RETRIES,
        "timeout": settings.FOOD_VISION_TIMEOUT_SECONDS,
    }
    if settings.FOOD_VISION_EFFECTIVE_BASE_URL:
        kwargs["base_url"] = settings.FOOD_VISION_EFFECTIVE_BASE_URL

    return OpenAI(**kwargs)


def _estimate_with_vision_client(client: Any, *, data_url: str) -> str:
    if getattr(getattr(client, "chat", None), "completions", None) is not None:
        return _estimate_with_chat_client(client, data_url=data_url)
    if getattr(client, "responses", None) is not None:
        return _estimate_with_responses_client(client, data_url=data_url)
    raise DietImageEstimatorError("food vision client does not support multimodal calls")


def _estimate_with_chat_client(client: Any, *, data_url: str) -> str:
    try:
        response = client.chat.completions.create(
            model=settings.FOOD_VISION_EFFECTIVE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FOOD_IMAGE_ESTIMATE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=settings.FOOD_VISION_MAX_OUTPUT_TOKENS,
            temperature=0,
        )
    except Exception as exc:
        raise DietImageEstimatorError("food vision chat model call failed") from exc

    return _extract_response_text(response)


def _estimate_with_responses_client(client: Any, *, data_url: str) -> str:
    try:
        response = client.responses.create(
            model=settings.FOOD_VISION_EFFECTIVE_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": FOOD_IMAGE_ESTIMATE_PROMPT},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
        )
    except Exception as exc:
        raise DietImageEstimatorError("food vision model call failed") from exc

    return _extract_response_text(response)


def _estimate_with_agent_llm(*, data_url: str) -> str:
    try:
        from langchain_core.messages import HumanMessage

        response = get_food_agent_llm().invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": FOOD_IMAGE_ESTIMATE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]
                )
            ]
        )
    except Exception as exc:
        raise DietImageEstimatorError("agent vision model call failed") from exc

    return _extract_response_text(response)


@lru_cache(maxsize=1)
def get_food_agent_llm() -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise DietImageEstimatorError("langchain_openai package is not installed") from exc

    return ChatOpenAI(
        api_key=settings.AGENT_LLM_EFFECTIVE_API_KEY,
        base_url=settings.AGENT_LLM_BASE_URL,
        default_headers=settings.OPENAI_COMPAT_DEFAULT_HEADERS,
        max_retries=settings.FOOD_VISION_MAX_RETRIES,
        model=settings.FOOD_VISION_EFFECTIVE_MODEL,
        temperature=0,
        timeout=settings.FOOD_VISION_TIMEOUT_SECONDS,
    )


def _prepare_image_for_model(*, image_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_bytes, mime_type

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            elif image.mode == "L":
                image = image.convert("RGB")
            image.thumbnail((768, 768))
            output = BytesIO()
            image.save(output, format="JPEG", quality=70, optimize=True)
            return output.getvalue(), "image/jpeg"
    except Exception:
        return image_bytes, mime_type


def normalize_food_estimate_payload(payload: dict[str, Any]) -> FoodImageEstimateResult:
    raw_items = payload.get("items")
    items = [_normalize_item(item) for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
    total = _total_from_items(items)

    return FoodImageEstimateResult(
        items=items,
        total=total,
        need_user_confirmation=True,
        warning=FOOD_ESTIMATE_WARNING,
    )


def _normalize_item(item: dict[str, Any]) -> FoodEstimateItem:
    assumptions = item.get("assumptions")
    if isinstance(assumptions, list):
        normalized_assumptions = [str(value).strip() for value in assumptions if str(value).strip()]
    elif assumptions:
        normalized_assumptions = [str(assumptions).strip()]
    else:
        normalized_assumptions = []

    return FoodEstimateItem(
        name=str(item.get("name") or "").strip(),
        estimated_weight_g=_number(item.get("estimated_weight_g")),
        estimated_kcal=_number(item.get("estimated_kcal")),
        min_kcal=_number(item.get("min_kcal")),
        max_kcal=_number(item.get("max_kcal")),
        protein_g=_number(item.get("protein_g")),
        fat_g=_number(item.get("fat_g")),
        carbs_g=_number(item.get("carbs_g")),
        confidence=max(0, min(1, _number(item.get("confidence")))),
        assumptions=normalized_assumptions,
        source="ai_estimated",
    )


def _total_from_items(items: list[FoodEstimateItem]) -> FoodEstimateTotal:
    return FoodEstimateTotal(
        estimated_kcal=_sum(items, "estimated_kcal"),
        min_kcal=_sum(items, "min_kcal"),
        max_kcal=_sum(items, "max_kcal"),
        protein_g=_sum(items, "protein_g"),
        fat_g=_sum(items, "fat_g"),
        carbs_g=_sum(items, "carbs_g"),
    )


def _sum(items: list[FoodEstimateItem], field: str) -> float:
    return _round_one_decimal(sum(float(getattr(item, field)) for item in items))


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, _round_one_decimal(number))


def _round_one_decimal(number: float) -> float:
    try:
        rounded = Decimal(str(number)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return 0
    return float(rounded)


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    content = getattr(response, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts = _extract_text_parts(content)
        if text_parts:
            return "\n".join(text_parts)

    if isinstance(response, str):
        return response

    choices = getattr(response, "choices", None)
    if isinstance(choices, list):
        for choice in choices:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")

            # GLM-4.6V: check content first
            message_content = None
            if message is not None:
                message_content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
            if isinstance(message_content, str) and message_content.strip():
                return message_content
            if isinstance(message_content, list):
                text_parts = _extract_text_parts(message_content)
                if text_parts:
                    return "\n".join(text_parts)

            # GLM-4.6V: fallback to reasoning_content
            reasoning = None
            if message is not None:
                reasoning = getattr(message, "reasoning_content", None) if not isinstance(message, dict) else message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning

    output = getattr(response, "output", None)
    if isinstance(output, list):
        text_parts: list[str] = []
        for item in output:
            item_content = getattr(item, "content", None)
            if not isinstance(item_content, list):
                item_content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(item_content, list):
                continue
            text_parts.extend(_extract_text_parts(item_content))
        if text_parts:
            return "\n".join(text_parts)

    return str(response)


def _extract_text_parts(parts: list[Any]) -> list[str]:
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is None and isinstance(part, dict):
            text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return text_parts
