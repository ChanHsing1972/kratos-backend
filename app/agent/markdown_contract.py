"""Shared Markdown output contract for user-visible Agent answers."""

from __future__ import annotations

import re


STANDARD_MARKDOWN_OUTPUT_PROMPT = """
输出格式必须是标准 Markdown：
- 只输出用户可见答案，不输出内部任务编号、JSON 合约、调试信息或自定义标签。
- 使用自然段、标准标题、标准有序/无序列表、标准引用、标准 GFM 表格和 fenced code block。
- 标题必须独占一行，格式为 `## 标题` 或 `### 标题`；不要把标题粘在段落或列表后面。
- 列表只使用 `- 内容` 或 `1. 内容`；不要使用伪列表、装饰符、重复项目符号或用 `>` 做装饰。
- 表格必须是标准 GFM 多行表格：表头、分隔行、数据行分别独占一行，且列数一致。
- 代码必须放在 fenced code block 中，并在开头注明语言，例如 ```python。
- 不要混用 HTML、XML、自定义标签、`<br>`、内联样式或无法由标准 Markdown 渲染器稳定解析的格式。
- 不要输出未闭合的加粗、斜体、链接、表格或代码围栏；不确定时使用普通文本。
- 控制嵌套深度，常规回答以 2 到 4 个短小段落或列表块为宜。
- 中文回答保持清晰克制，避免过度装饰符、emoji、奇怪符号和重复分隔线。
""".strip()


def finalize_markdown_response(response_text: str) -> str:
    """Apply generic Markdown-safe finalization and repair common gluing errors."""

    text = str(response_text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s*<br\s*/?>\s*", "、", text, flags=re.IGNORECASE)
    lines = _repair_markdown_lines([line.rstrip() for line in text.split("\n")])
    return "\n".join(lines).strip()


def _repair_markdown_lines(lines: list[str]) -> list[str]:
    repaired: list[str] = []
    in_code_fence = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if re.match(r"^ {0,3}(`{3,}|~{3,})", line):
            in_code_fence = not in_code_fence
            repaired.append(line)
            continue
        if in_code_fence:
            repaired.append(line)
            continue

        split_heading = _split_glued_heading(line)
        normalized_lines = (
            split_heading
            if split_heading is not None
            else [_repair_markdown_line(line)]
        )
        for normalized in normalized_lines:
            _append_with_markdown_spacing(repaired, normalized)

    return _collapse_blank_lines(repaired)


def _repair_markdown_line(line: str) -> str:
    line = _repair_heading_space(line)
    line = _repair_list_marker(line)
    if _is_table_line(line):
        return _repair_table_row(line)
    return line


def _split_glued_heading(line: str) -> list[str] | None:
    table_match = re.match(r"^(#{1,6})\s*([^|\n]{1,48})\|(.+)$", line)
    if table_match:
        hashes, title, table_tail = table_match.groups()
        return [
            f"{hashes} {title.strip()}",
            "",
            _repair_table_row(f"| {table_tail.strip()}"),
        ]

    list_match = re.match(r"^(#{1,6})\s*(.{2,48}?)(?<!\d)-\s*(?!\d)(\S.*)$", line)
    if list_match:
        hashes, title, item = list_match.groups()
        return [
            f"{hashes} {title.strip()}",
            "",
            f"- {item.strip()}",
        ]
    overlong_heading = _split_overlong_heading(line)
    if overlong_heading is not None:
        return overlong_heading
    return None


def _split_overlong_heading(line: str) -> list[str] | None:
    match = re.match(r"^(#{1,6})\s*(\S.{24,})$", _repair_heading_space(line))
    if not match:
        return None

    hashes, content = match.groups()
    if "|" in content or content.startswith("- "):
        return None
    title_match = re.match(
        r"^((?:今日|今天|本次|本周|一周|每周|周期)?"
        r"(?:肩部|腹部|上肢|下肢|全身|胸部|背部|腿部|臀腿|核心|恢复|饮食|营养|增肌|减脂|力量|康复|周期|长期)*"
        r"(?:训练建议与安排|训练计划|训练安排|训练建议|训练|饮食建议|营养建议|计划|建议|安排))(.+)$",
        content,
    )
    if not title_match:
        title_match = re.match(
            r"^(.{4,48}?(?:训练建议与安排|训练计划|训练安排|训练建议|饮食建议|营养建议|计划|建议|安排))"
            r"((?:根据|结合|下面|以下|注意[:：]?|如有).+)$",
            content,
        )
    if not title_match:
        return None

    title, suffix = title_match.groups()
    suffix = suffix.lstrip(" ，,。:：")
    if not suffix:
        return None
    if not re.match(r"^(根据|结合|下面|以下|注意[:：]?|如有|若|如果)", suffix):
        return None
    if len(title) < 4:
        return None
    return [
        f"{hashes} {title.strip()}",
        "",
        suffix,
    ]


def _repair_heading_space(line: str) -> str:
    return re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)


def _repair_list_marker(line: str) -> str:
    return re.sub(r"^([ \t]*)-([^\s-])", r"\1- \2", line)


def _repair_table_row(line: str) -> str:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if not cells:
        return line
    if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
        cells = ["---" for _ in cells]
    return "| " + " | ".join(cells) + " |"


def _append_with_markdown_spacing(lines: list[str], line: str) -> None:
    is_heading = _is_heading_line(line)
    is_table = _is_table_line(line)
    previous = _last_nonempty(lines)
    if is_heading and previous:
        lines.append("")
    elif is_table and previous and not _is_table_line(previous):
        lines.append("")
    elif previous and _is_heading_line(previous) and line.strip():
        lines.append("")
    lines.append(line)


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    collapsed: list[str] = []
    for line in lines:
        if not line.strip() and collapsed and not collapsed[-1].strip():
            continue
        collapsed.append(line)
    return collapsed


def _last_nonempty(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if line.strip():
            return line
    return None


def _is_heading_line(line: str) -> bool:
    return bool(re.match(r"^(#{1,6})\s+\S", line))


def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and "|" in line.strip()[1:]


def markdown_contract_violations(response_text: str) -> list[str]:
    """Return hard Markdown contract violations without rewriting content."""

    text = str(response_text or "")
    violations: list[str] = []

    if re.search(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^>]*)?>", text):
        violations.append("最终回复包含 HTML 或自定义标签，需要改为标准 Markdown。")

    open_fence = _open_code_fence(text)
    if open_fence is not None:
        violations.append("最终回复存在未闭合的 fenced code block。")

    violations.extend(_table_contract_violations(text))

    return violations


def _table_contract_violations(markdown: str) -> list[str]:
    table_blocks = _table_blocks(markdown)
    for block in table_blocks:
        rows = [_table_cells(line) for line in block]
        rows = [row for row in rows if len(row) >= 2]
        if not rows:
            continue
        expected_columns = len(rows[0])
        has_separator = len(rows) >= 2 and all(_is_table_separator_cell(cell) for cell in rows[1])
        if not has_separator:
            return ["最终回复包含不完整 Markdown 表格，需要表头、分隔行和数据行。"]
        if len(rows) < 3:
            return ["最终回复包含不完整 Markdown 表格，需要表头、分隔行和数据行。"]
        if any(len(row) != expected_columns for row in rows):
            return ["最终回复包含列数不一致的 Markdown 表格，需要修正为标准 GFM 表格。"]
    return []


def _table_blocks(markdown: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    in_code_fence = False
    for raw_line in str(markdown or "").splitlines():
        line = raw_line.strip()
        if re.match(r"^ {0,3}(`{3,}|~{3,})", raw_line):
            in_code_fence = not in_code_fence
            if current:
                blocks.append(current)
                current = []
            continue
        if in_code_fence:
            continue
        if _is_table_line(line):
            current.append(line)
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _table_cells(line: str) -> list[str]:
    if not _is_table_line(line):
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator_cell(cell: str) -> bool:
    return bool(re.fullmatch(r":?-{2,}:?", cell.strip()))


def _open_code_fence(markdown: str) -> str | None:
    open_marker: str | None = None
    for line in markdown.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if not match:
            continue
        marker = match.group(1)
        if open_marker is None:
            open_marker = marker
            continue
        if marker[0] == open_marker[0] and len(marker) >= len(open_marker):
            open_marker = None
    return open_marker
