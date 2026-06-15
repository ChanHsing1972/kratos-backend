from uuid import uuid4

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.skill import Skill, UserSkill
from app.models.user import User
from app.schemas.skill import SkillCreate, SkillUpdate


BASELINE_UTILITY_TOOLS = {
    "weather_fitness_advisor",
    "heweather_geo_lookup",
    "heweather_weather",
    "amap_geocode",
    "amap_place_search_around",
    "amap_distance",
    "amap_walking_route",
    "amap_driving_route",
    "amap_bicycling_route",
    "amap_transit_route",
    "place_navigation_advisor",
    "tavily_search",
}


BUILTIN_SKILLS: list[dict] = [
    {
        "name": "减脂教练",
        "slug": "builtin-fat-loss-coach",
        "description": "面向减脂、控卡、提升日常消耗的轻量训练与饮食策略。",
        "applicable_scenarios": "用户目标是减脂、降低体脂、控制热量、提高运动消耗或需要低门槛执行方案。",
        "prompt_snippet": (
            "以减脂教练身份回答：优先建立温和热量缺口，强调蛋白质、蔬菜、睡眠和可持续训练；"
            "训练建议以低风险有氧、力量保肌和日常活动量为核心。"
        ),
        "available_tools": [
            "calculate_bmr",
            "calculate_calories_burned",
            "calculate_workout_volume",
            "diet_plan_generator",
            "weather_fitness_advisor",
            "place_navigation_advisor",
            "running_route_advisor",
        ],
        "output_format": "先给今日行动，再给训练/饮食安排，最后列出风险边界和复盘指标。",
        "forbidden_rules": "不要建议极端节食、脱水、催吐、短期暴瘦或忽视疼痛硬练。",
    },
    {
        "name": "增肌教练",
        "slug": "builtin-muscle-gain-coach",
        "description": "面向增肌、力量进阶和训练容量管理的渐进超负荷策略。",
        "applicable_scenarios": "用户目标是增肌、提高力量、安排分化训练、估算训练重量或优化蛋白质摄入。",
        "prompt_snippet": (
            "以增肌教练身份回答：围绕渐进超负荷、足量蛋白、动作质量和恢复周期制定建议；"
            "说明组数、次数、强度区间和进阶条件。"
        ),
        "available_tools": [
            "estimate_1rm",
            "calculate_bmr",
            "calculate_workout_volume",
            "diet_plan_generator",
            "musclewiki_api",
            "rapidapi_bodyparts",
        ],
        "output_format": "用训练日安排、动作参数、营养目标、进阶规则四段组织。",
        "forbidden_rules": "不要鼓励带伤冲重量、无保护极限测试、滥用药物或长期忽略恢复。",
    },
    {
        "name": "康复安全教练",
        "slug": "builtin-rehab-safety-coach",
        "description": "面向疼痛、疲劳、伤病史和训练安全边界的保守策略。",
        "applicable_scenarios": "用户提到疼痛、不适、受伤、术后、极度疲劳、睡眠不足或需要替代训练。",
        "prompt_snippet": (
            "以康复安全教练身份回答：先做风险分层，再给降级动作和停止条件；"
            "疼痛、麻木、头晕、急性损伤等信号优先安全处理。"
        ),
        "available_tools": [
            "pain_safety_gate",
            "calculate_workout_volume",
            "weather_fitness_advisor",
        ],
        "output_format": "先判断是否能练，再给替代动作，最后给停止条件和就医提醒。",
        "forbidden_rules": "不要给出诊断结论；不要要求用户忍痛训练；不要替代医生、康复师或急救建议。",
    },
]


def seed_builtin_skills(db: Session) -> None:
    changed = False
    for item in BUILTIN_SKILLS:
        skill = db.query(Skill).filter(Skill.slug == item["slug"]).first()
        payload = {
            **item,
            "definition": build_skill_definition(item),
            "is_builtin": True,
            "is_public": True,
            "owner_user_id": None,
        }
        if skill is None:
            db.add(Skill(**payload))
            changed = True
            continue

        for field, value in payload.items():
            if getattr(skill, field) != value:
                setattr(skill, field, value)
                changed = True

    if changed:
        db.commit()


def list_skills_for_user(db: Session, user_id: int) -> list[dict]:
    seed_builtin_skills(db)
    enabled_map = _enabled_skill_map(db, user_id)
    skills = (
        db.query(Skill)
        .filter(or_(Skill.is_builtin.is_(True), Skill.owner_user_id == user_id))
        .order_by(Skill.is_builtin.desc(), Skill.created_at.desc(), Skill.id.desc())
        .all()
    )
    return [_skill_to_response(skill, enabled_map.get(skill.id, False)) for skill in skills]


def list_my_skills(db: Session, user_id: int) -> list[dict]:
    return [
        item
        for item in list_skills_for_user(db, user_id)
        if item["enabled"] or item["owner_user_id"] == user_id
    ]


def get_enabled_skills_for_user(db: Session, user_id: int) -> list[Skill]:
    seed_builtin_skills(db)
    return (
        db.query(Skill)
        .join(UserSkill, UserSkill.skill_id == Skill.id)
        .filter(
            UserSkill.user_id == user_id,
            UserSkill.enabled.is_(True),
            or_(Skill.is_builtin.is_(True), Skill.owner_user_id == user_id),
        )
        .order_by(Skill.is_builtin.desc(), Skill.created_at.desc(), Skill.id.desc())
        .all()
    )


def get_accessible_skill(db: Session, skill_id: int, user_id: int) -> Skill | None:
    seed_builtin_skills(db)
    return (
        db.query(Skill)
        .filter(
            Skill.id == skill_id,
            or_(Skill.is_builtin.is_(True), Skill.owner_user_id == user_id),
        )
        .first()
    )


def get_editable_skill(db: Session, skill_id: int, user_id: int) -> Skill | None:
    return (
        db.query(Skill)
        .filter(
            Skill.id == skill_id,
            Skill.owner_user_id == user_id,
            Skill.is_builtin.is_(False),
        )
        .first()
    )


def create_skill(db: Session, user: User, skill_in: SkillCreate) -> dict:
    data = skill_in.model_dump()
    data["available_tools"] = normalize_tool_names(data.get("available_tools"))
    if not data.get("definition"):
        data["definition"] = build_skill_definition(data)

    skill = Skill(
        owner_user_id=user.id,
        slug=f"custom-{user.id}-{uuid4().hex[:10]}",
        is_builtin=False,
        is_public=False,
        **data,
    )
    db.add(skill)
    db.flush()
    db.add(UserSkill(user_id=user.id, skill_id=skill.id, enabled=True))
    db.commit()
    db.refresh(skill)
    return _skill_to_response(skill, True)


def update_skill(
    db: Session,
    skill: Skill,
    skill_in: SkillUpdate,
    enabled: bool,
) -> dict:
    data = skill_in.to_update_dict()
    if "available_tools" in data:
        data["available_tools"] = normalize_tool_names(data.get("available_tools"))

    for field, value in data.items():
        setattr(skill, field, value)

    if "definition" not in data:
        skill.definition = build_skill_definition(skill_to_prompt_payload(skill))

    db.add(skill)
    db.commit()
    db.refresh(skill)
    return _skill_to_response(skill, enabled)


def set_user_skill_enabled(
    db: Session,
    user_id: int,
    skill: Skill,
    enabled: bool,
) -> dict:
    binding = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == user_id, UserSkill.skill_id == skill.id)
        .first()
    )
    if binding is None:
        binding = UserSkill(user_id=user_id, skill_id=skill.id, enabled=enabled)
        db.add(binding)
    else:
        binding.enabled = enabled
        db.add(binding)

    db.commit()
    db.refresh(skill)
    return _skill_to_response(skill, enabled)


def delete_skill(db: Session, skill: Skill) -> None:
    db.delete(skill)
    db.commit()


def is_skill_enabled_for_user(db: Session, user_id: int, skill_id: int) -> bool:
    binding = (
        db.query(UserSkill)
        .filter(UserSkill.user_id == user_id, UserSkill.skill_id == skill_id)
        .first()
    )
    return bool(binding and binding.enabled)


def allowed_tool_names(skills: list[Skill]) -> set[str]:
    names: set[str] = set()
    for skill in skills:
        names.update(normalize_tool_names(skill.available_tools))
    if skills:
        names.update(BASELINE_UTILITY_TOOLS)
    return names


def skill_to_prompt_payload(skill: Skill) -> dict:
    return {
        "id": skill.id,
        "name": skill.name,
        "applicable_scenarios": skill.applicable_scenarios,
        "prompt_snippet": skill.prompt_snippet,
        "available_tools": normalize_tool_names(skill.available_tools),
        "output_format": skill.output_format,
        "forbidden_rules": skill.forbidden_rules,
        "definition": skill.definition,
    }


def build_skill_definition(data: dict) -> str:
    tools = normalize_tool_names(data.get("available_tools"))
    tool_lines = "\n".join(f"  - {tool}" for tool in tools) or "  - none"
    name = data.get("name") or ""
    description = data.get("description") or ""
    scenarios = data.get("applicable_scenarios") or ""
    prompt = data.get("prompt_snippet") or ""
    output_format = data.get("output_format") or ""
    forbidden_rules = data.get("forbidden_rules") or ""
    return f"""---
name: {name}
description: {description}
applicable_scenarios: {scenarios}
available_tools:
{tool_lines}
---
# 何时使用
当用户请求符合 applicable_scenarios，或当前对话明确需要本 Skill 的专业视角时使用。Skill 只改变推理策略、风险边界和工具优先级，不代表工具一定已经执行。

# 工作流
1. 先识别用户真正目标、约束、地点、时间和风险信号。
2. 区分已确认资料、用户本轮新信息和外部实时信息。
3. 需要天气、新闻、地点、导航等实时信息时，先调用对应工具；最终回答只能引用工具结果。
4. 给出用户今天能直接执行的建议，同时说明不确定性和下一步。

# 策略提示
{prompt}

# 推荐工具
available_tools 是本 Skill 的优先工具集合；基础天气、搜索、地图和导航工具仍可用于普通信息查询。不要因为 Skill 未列出某个基础工具就拒绝用户的天气、地点或导航需求。

# 输出格式
{output_format}

# 安全边界
{forbidden_rules}

# 禁止事项
- 不要编造未读取的用户资料。
- 不要声称已经调用未出现在轨迹中的工具。
- 不要把外部实时信息写成确定事实，除非它来自本轮工具结果。
"""


def normalize_tool_names(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = value.replace("，", ",").replace("\n", ",").split(",")
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return normalized


def _enabled_skill_map(db: Session, user_id: int) -> dict[int, bool]:
    bindings = db.query(UserSkill).filter(UserSkill.user_id == user_id).all()
    return {binding.skill_id: binding.enabled for binding in bindings}


def _skill_to_response(skill: Skill, enabled: bool) -> dict:
    return {
        "id": skill.id,
        "name": skill.name,
        "slug": skill.slug,
        "description": skill.description,
        "applicable_scenarios": skill.applicable_scenarios,
        "prompt_snippet": skill.prompt_snippet,
        "available_tools": normalize_tool_names(skill.available_tools),
        "output_format": skill.output_format,
        "forbidden_rules": skill.forbidden_rules,
        "definition": skill.definition,
        "owner_user_id": skill.owner_user_id,
        "is_builtin": skill.is_builtin,
        "is_public": skill.is_public,
        "enabled": enabled,
        "source": "builtin" if skill.is_builtin else "custom",
        "created_at": skill.created_at,
        "updated_at": skill.updated_at,
    }
