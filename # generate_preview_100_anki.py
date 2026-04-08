import json
import time
import html
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI()

MODEL = "gpt-5.4-mini"

INPUT_FILE = "entries.json"
PROGRESS_FILE = "progress.json"
CARDS_JSON_FILE = "cards_all.json"
ANKI_TSV_FILE = "anki_all.tsv"

RUN_SIZE = 5000
BATCH_SIZE = 10
SLEEP_SECONDS = 1.1


SYSTEM_PROMPT = """
你是一位精通德语、中文和英语的语言学专家，专门为一位在德国生活和学习法律的中国学生制作 Anki 词汇卡片。

【项目目标】
目标不是机械处理整本词典，而是从词典中筛选出最值得学习的内容，帮助学习者达到：
1. 在德国日常生活中理解和表达基本无障碍；
2. 在一般阅读中基本无障碍；
3. 重点掌握 C1-C2 日常高频层与中频识别层；
4. 对 A1-B2 基础词，默认跳过，除非其固定搭配、特殊用法或正式语体延伸具有较高学习价值。

【你的任务】
对于我给出的每一批 Duden DaF 词典词条原文，判断每个词条应当：
- 跳过（skip）
- 生成高频主动卡（aktiv）
- 生成中频识别卡（passiv）
- 只生成基础词搭配卡（kollokation）

【筛选原则】
一、跳过（skip）
- A1-B2 基础词且无高价值高级搭配/特殊用法
- 极低频、过时、冷僻、过于专业化、对“德国日常生活 + 一般阅读”目标帮助很小的词
- 不值得为其单独制作 Anki 卡片的词

二、高频卡（aktiv）
适用于：
- 值得主动掌握并在生活、社会讨论、大学环境、制度生活、一般阅读中主动使用的词
- 日常高频或准高频、且具备较强表达价值的 C1-C2 层词汇

三、中频卡（passiv）
适用于：
- 阅读中会遇到、生活中可能见到、但不需要强求主动说出的表达
- 应以“看见能懂”为目标

四、基础词搭配卡（kollokation）
适用于：
- 该词本身过于基础，不值得做完整词汇卡
- 但它存在高价值固定搭配、正式语体用法、易误用结构或学习者常掌握不牢的表达
- 此时不要输出完整词条卡，只输出搭配项

【统一输出规则】
你必须只输出一个 JSON 对象：
{
  "cards": [
    {
      "wort": "德语词条",
      "typ": "aktiv | passiv | kollokation",
      "wortart": "词性或 null",
      "genus": "阳/阴/中或 null",
      "zh": "中文释义或 null",
      "en": "英文对应或 null",
      "deutsch_def": "简明德语释义或 null",
      "vergleich": "辨析或 null",
      "beispiele": [
        {"satz": "德语例句", "zh": "中文翻译"}
      ],
      "kollokationen": ["搭配1", "搭配2"],
      "hinweis": "说明或 null",
      "items": [
        {"搭配": "固定搭配", "zh": "中文意思", "beispiel": "德语例句"}
      ]
    }
  ]
}

【字段填写规则】
- aktiv：
  - typ = "aktiv"
  - 必须填写：wort, wortart, zh, en, deutsch_def
  - beispiele 必须恰好有 2 个
  - kollokationen 至少给 2 个，若确实没有可给空数组
  - genus 仅名词时填写，否则为 null
  - vergleich / hinweis 可为 null
  - items 必须为空数组

- passiv：
  - typ = "passiv"
  - 必须填写：wort, wortart, zh, en, deutsch_def
  - beispiele 必须恰好有 1 个
  - genus 仅名词时填写，否则为 null
  - vergleich 通常为 null
  - kollokationen 通常为空数组
  - items 必须为空数组
  - hinweis 可为 null

- kollokation：
  - typ = "kollokation"
  - 必须填写：wort, items
  - items 至少 1 个
  - 其余字段统一填 null 或空数组
  - 不要输出完整词义卡

【质量标准】
- 中文释义必须自然、精准，不要僵硬直译
- 英文对应要尽量贴切
- 德语例句必须自然、地道、符合真实语境
- 优先考虑真实德国生活、大学场景、公共制度环境、社会讨论、常见阅读文本中的使用价值
- 固定搭配尽量注明介词和格要求，如：sich verlassen auf + Akk.
- 如果一个词有多个常用义项，在 zh 字段中用 ①② 标注
- 不要为了凑数量保留低价值词条
- 被完全跳过的词条不要出现在结果中

【硬性要求】
- 只输出 JSON 对象，不要输出任何其他文字
- 不要输出 markdown 代码块
- 所有卡片都必须遵守同一个字段集合
- 不要使用不同对象结构
"""


class Beispiel(BaseModel):
    satz: str
    zh: str


class KollokationItem(BaseModel):
    phrase: str = Field(alias="搭配")
    zh: str
    beispiel: str


class CardItem(BaseModel):
    wort: str
    typ: Literal["aktiv", "passiv", "kollokation"]

    wortart: Optional[str] = None
    genus: Optional[str] = None

    zh: Optional[str] = None
    en: Optional[str] = None
    deutsch_def: Optional[str] = None
    vergleich: Optional[str] = None
    beispiele: List[Beispiel] = Field(default_factory=list)
    kollokationen: List[str] = Field(default_factory=list)
    hinweis: Optional[str] = None

    items: List[KollokationItem] = Field(default_factory=list)


class BatchResult(BaseModel):
    cards: List[CardItem]


def esc(x: Optional[str]) -> str:
    return html.escape(x or "")


def load_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_entries(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_input_text(entries_batch: list) -> str:
    parts = ["请处理以下 Duden DaF 词典词条原文：\n"]
    for entry in entries_batch:
        entry_id = entry.get("entry_id", "")
        lemma = entry.get("lemma", "")
        raw_text = entry.get("raw_text", "")
        parts.append(f"---\nentry_id: {entry_id}\nlemma: {lemma}\n{raw_text}\n")
    return "\n".join(parts)


def model_dump_card(card: CardItem) -> dict:
    return card.model_dump(by_alias=True, exclude_none=False)


def process_batch(entries_batch: list, batch_id: int) -> list:
    input_text = build_input_text(entries_batch)

    try:
        response = client.responses.parse(
            model=MODEL,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ],
            text_format=BatchResult
        )

        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("模型未返回可解析的结构化结果")

        return [model_dump_card(card) for card in parsed.cards]

    except Exception as e:
        error_path = f"error_batch_{batch_id}.json"
        save_json(error_path, {
            "batch_id": batch_id,
            "error": str(e),
            "entries_batch": entries_batch
        })
        print(f"批次 {batch_id} 出错，已写入 {error_path}")
        return []


def color_for_type(card_type: str) -> dict:
    palette = {
        "aktiv": {"accent": "#2D6A4F", "soft": "#EEF6F1", "label": "主动高频"},
        "passiv": {"accent": "#1D3557", "soft": "#EEF3F8", "label": "被动识别"},
        "kollokation": {"accent": "#7B2CBF", "soft": "#F4EEFB", "label": "固定搭配"}
    }
    return palette.get(card_type, {"accent": "#444444", "soft": "#F6F6F6", "label": "词卡"})


def render_badge(text: str, bg: str, fg: str = "#ffffff") -> str:
    return (
        f"<span style=\"display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:12px;font-weight:700;letter-spacing:.3px;\">"
        f"{esc(text)}</span>"
    )


def render_section_title(title: str, accent: str) -> str:
    return (
        f"<div style=\"font-size:13px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;"
        f"color:{accent};margin:18px 0 8px 0;\">{esc(title)}</div>"
    )


def render_example_block(example_de: str, example_zh: str, accent: str) -> str:
    return f"""
    <div style="background:#fff;border:1px solid #e9ecef;border-left:4px solid {accent};
                border-radius:10px;padding:12px 14px;margin:10px 0;">
        <div style="font-size:15px;line-height:1.7;color:#1f2937;font-style:italic;">
            {esc(example_de)}
        </div>
        <div style="font-size:13px;line-height:1.7;color:#6b7280;margin-top:6px;">
            {esc(example_zh)}
        </div>
    </div>
    """


def render_list_items(items: List[str], accent: str) -> str:
    if not items:
        return "<div style='color:#888;font-size:13px;'>—</div>"
    lis = []
    for item in items:
        lis.append(
            f"<li style=\"margin:6px 0;line-height:1.7;color:#374151;\">"
            f"<span style=\"color:{accent};font-weight:700;\">•</span> {esc(item)}</li>"
        )
    return f"<ul style=\"padding-left:18px;margin:6px 0 0 0;\">{''.join(lis)}</ul>"


def build_front_html(card: dict) -> str:
    p = color_for_type(card["typ"])
    accent = p["accent"]
    soft = p["soft"]
    label = p["label"]

    if card["typ"] == "aktiv":
        head = esc(card.get("zh"))
        sub = f"{esc(card.get('wortart'))}" + (f" · {esc(card.get('genus'))}" if card.get("genus") else "")
        prompt = "请根据中文义项回忆德语词、典型搭配与常见用法。"
    elif card["typ"] == "passiv":
        head = esc(card.get("wort"))
        sub = esc(card.get("wortart"))
        prompt = "看到这个词时，先判断大致意思，再回忆使用场景。"
    else:
        head = esc(card.get("wort"))
        sub = "基础词的高价值搭配"
        prompt = "回忆这个基础词最值得掌握的固定搭配。"

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;max-width:680px;margin:0 auto;padding:22px;background:linear-gradient(180deg,#ffffff 0%,#fcfcfd 100%);color:#1f2937;">
        <div style="border:1px solid #e5e7eb;border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.06);">
            <div style="background:{soft};padding:18px 20px;border-bottom:1px solid #e5e7eb;">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                    {render_badge(label, accent)}
                    <div style="font-size:12px;color:#6b7280;">German Lexicon Project</div>
                </div>
                <div style="font-size:28px;font-weight:800;line-height:1.25;color:#111827;margin-top:14px;">{head}</div>
                <div style="font-size:14px;color:#6b7280;margin-top:8px;">{sub}</div>
            </div>
            <div style="padding:22px 20px 24px 20px;">
                <div style="font-size:15px;line-height:1.8;color:#374151;background:#fafafa;border-radius:12px;padding:14px 16px;border:1px dashed #d1d5db;">
                    {esc(prompt)}
                </div>
            </div>
        </div>
    </div>
    """


def build_back_html(card: dict) -> str:
    p = color_for_type(card["typ"])
    accent = p["accent"]
    soft = p["soft"]
    label = p["label"]

    if card["typ"] == "aktiv":
        examples_html = "".join(
            render_example_block(ex["satz"], ex["zh"], accent)
            for ex in card.get("beispiele", [])
        )

        compare_html = ""
        if card.get("vergleich"):
            compare_html = f"""
            {render_section_title("辨析", accent)}
            <div style="background:#fff;border:1px solid #e9ecef;border-radius:10px;padding:12px 14px;font-size:14px;line-height:1.8;color:#374151;">
                {esc(card["vergleich"])}
            </div>
            """

        note_html = ""
        if card.get("hinweis"):
            note_html = f"""
            {render_section_title("注意事项", accent)}
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;font-size:14px;line-height:1.8;color:#7c2d12;">
                {esc(card["hinweis"])}
            </div>
            """

        genus_line = (
            f"<div style='margin-top:6px;color:#6b7280;font-size:14px;'>词性：{esc(card.get('wortart'))} · 性：{esc(card.get('genus', '—'))}</div>"
            if card.get("genus")
            else f"<div style='margin-top:6px;color:#6b7280;font-size:14px;'>词性：{esc(card.get('wortart'))}</div>"
        )

        return f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;max-width:760px;margin:0 auto;padding:22px;background:linear-gradient(180deg,#ffffff 0%,#fcfcfd 100%);color:#1f2937;">
            <div style="border:1px solid #e5e7eb;border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.06);">
                <div style="background:{soft};padding:18px 20px;border-bottom:1px solid #e5e7eb;">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                        {render_badge(label, accent)}
                        <div style="font-size:12px;color:#6b7280;">German ↔ Chinese ↔ English</div>
                    </div>
                    <div style="font-size:30px;font-weight:800;line-height:1.2;color:#111827;margin-top:14px;">{esc(card.get("wort"))}</div>
                    {genus_line}
                </div>

                <div style="padding:22px 20px 24px 20px;">
                    <div style="display:grid;grid-template-columns:1fr;gap:12px;">
                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">中文义项</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("zh"))}</div>
                        </div>

                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">English</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("en"))}</div>
                        </div>

                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">Deutsch erklärt</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("deutsch_def"))}</div>
                        </div>
                    </div>

                    {render_section_title("例句", accent)}
                    {examples_html}

                    {render_section_title("常见搭配", accent)}
                    <div style="background:#fff;border:1px solid #e9ecef;border-radius:10px;padding:12px 14px;">
                        {render_list_items(card.get("kollokationen", []), accent)}
                    </div>

                    {compare_html}
                    {note_html}
                </div>
            </div>
        </div>
        """

    if card["typ"] == "passiv":
        examples_html = "".join(
            render_example_block(ex["satz"], ex["zh"], accent)
            for ex in card.get("beispiele", [])
        )

        note_html = ""
        if card.get("hinweis"):
            note_html = f"""
            {render_section_title("理解难点", accent)}
            <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:12px 14px;font-size:14px;line-height:1.8;color:#7c2d12;">
                {esc(card["hinweis"])}
            </div>
            """

        return f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;max-width:760px;margin:0 auto;padding:22px;background:linear-gradient(180deg,#ffffff 0%,#fcfcfd 100%);color:#1f2937;">
            <div style="border:1px solid #e5e7eb;border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.06);">
                <div style="background:{soft};padding:18px 20px;border-bottom:1px solid #e5e7eb;">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                        {render_badge(label, accent)}
                        <div style="font-size:12px;color:#6b7280;">Recognition-oriented Card</div>
                    </div>
                    <div style="font-size:30px;font-weight:800;line-height:1.2;color:#111827;margin-top:14px;">{esc(card.get("wort"))}</div>
                    <div style="margin-top:6px;color:#6b7280;font-size:14px;">词性：{esc(card.get("wortart"))}</div>
                </div>

                <div style="padding:22px 20px 24px 20px;">
                    <div style="display:grid;grid-template-columns:1fr;gap:12px;">
                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">中文释义</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("zh"))}</div>
                        </div>

                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">English</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("en"))}</div>
                        </div>

                        <div style="background:#fff;border:1px solid #e9ecef;border-radius:12px;padding:14px 16px;">
                            <div style="font-size:13px;font-weight:700;color:{accent};margin-bottom:8px;">Deutsch erklärt</div>
                            <div style="font-size:15px;line-height:1.8;color:#111827;">{esc(card.get("deutsch_def"))}</div>
                        </div>
                    </div>

                    {render_section_title("例句", accent)}
                    {examples_html}
                    {note_html}
                </div>
            </div>
        </div>
        """

    items_html = ""
    for item in card.get("items", []):
        phrase = item.get("搭配", "")
        zh = item.get("zh", "")
        beispiel = item.get("beispiel", "")
        items_html += f"""
        <div style="background:#fff;border:1px solid #e9ecef;border-left:4px solid {accent};border-radius:10px;padding:14px 14px;margin:10px 0;">
            <div style="font-size:16px;font-weight:700;color:#111827;">{esc(phrase)}</div>
            <div style="font-size:14px;line-height:1.8;color:#4b5563;margin-top:6px;">{esc(zh)}</div>
            <div style="font-size:14px;line-height:1.8;color:#6b7280;margin-top:8px;font-style:italic;">
                {esc(beispiel)}
            </div>
        </div>
        """

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;max-width:760px;margin:0 auto;padding:22px;background:linear-gradient(180deg,#ffffff 0%,#fcfcfd 100%);color:#1f2937;">
        <div style="border:1px solid #e5e7eb;border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(0,0,0,.06);">
            <div style="background:{soft};padding:18px 20px;border-bottom:1px solid #e5e7eb;">
                <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
                    {render_badge(label, accent)}
                    <div style="font-size:12px;color:#6b7280;">Collocation-focused Card</div>
                </div>
                <div style="font-size:30px;font-weight:800;line-height:1.2;color:#111827;margin-top:14px;">{esc(card.get("wort"))}</div>
                <div style="margin-top:6px;color:#6b7280;font-size:14px;">基础词，但搭配值得专门掌握</div>
            </div>
            <div style="padding:22px 20px 24px 20px;">
                {items_html}
            </div>
        </div>
    </div>
    """


def make_tags(card: dict) -> str:
    base = ["GermanLexicon"]
    if card["typ"] == "aktiv":
        base += ["aktiv", "productive"]
    elif card["typ"] == "passiv":
        base += ["passiv", "recognition"]
    else:
        base += ["kollokation", "collocation"]
    return " ".join(base)


def to_anki_row(card: dict) -> list[str]:
    front = build_front_html(card)
    back = build_back_html(card)
    tags = make_tags(card)
    return [front, back, tags]


def export_tsv(cards: list, output_path: str):
    header = [
        "#separator:tab",
        "#html:true",
        "#tags column:3"
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for card in cards:
            row = to_anki_row(card)
            f.write("\t".join(s.replace("\t", " ").replace("\n", " ") for s in row) + "\n")


def init_progress(total_entries: int) -> dict:
    return {
        "next_index": 0,
        "total_entries": total_entries,
        "total_cards": 0,
        "runs_completed": 0,
        "finished": False
    }


def main():
    entries = load_entries(INPUT_FILE)
    total_entries = len(entries)

    progress = load_json(PROGRESS_FILE, init_progress(total_entries))
    all_cards = load_json(CARDS_JSON_FILE, [])

    next_index = progress.get("next_index", 0)

    if next_index >= total_entries:
        print("所有词条都已经处理完成。")
        print(f"累计卡片数：{len(all_cards)}")
        export_tsv(all_cards, ANKI_TSV_FILE)
        return

    end_index = min(next_index + RUN_SIZE, total_entries)
    current_slice = entries[next_index:end_index]

    print(f"本次将处理词条范围：{next_index} ~ {end_index - 1}")
    print(f"共 {len(current_slice)} 个词条，分批大小 {BATCH_SIZE}")

    total_batches = (len(current_slice) + BATCH_SIZE - 1) // BATCH_SIZE

    for local_i in range(0, len(current_slice), BATCH_SIZE):
        batch = current_slice[local_i:local_i + BATCH_SIZE]
        global_start = next_index + local_i
        batch_id = global_start // BATCH_SIZE + 1

        print(f"处理批次 {local_i // BATCH_SIZE + 1}/{total_batches} ...")
        cards = process_batch(batch, batch_id)
        all_cards.extend(cards)

        save_json(CARDS_JSON_FILE, all_cards)

        progress["total_cards"] = len(all_cards)
        save_json(PROGRESS_FILE, progress)

        time.sleep(SLEEP_SECONDS)

    progress["next_index"] = end_index
    progress["total_cards"] = len(all_cards)
    progress["runs_completed"] = progress.get("runs_completed", 0) + 1
    progress["finished"] = end_index >= total_entries

    save_json(PROGRESS_FILE, progress)
    save_json(CARDS_JSON_FILE, all_cards)
    export_tsv(all_cards, ANKI_TSV_FILE)

    print()
    print("本轮完成。")
    print(f"已处理到索引：{progress['next_index']} / {total_entries}")
    print(f"累计卡片数：{len(all_cards)}")
    print(f"JSON 文件：{CARDS_JSON_FILE}")
    print(f"Anki 文件：{ANKI_TSV_FILE}")

    if progress["finished"]:
        print("全部词条处理完成。")
    else:
        print("下次再次运行同一脚本，会自动继续处理后 500 个词条。")


if __name__ == "__main__":
    main()