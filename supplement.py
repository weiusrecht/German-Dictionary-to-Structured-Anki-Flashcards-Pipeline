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
PATCH_JSON_FILE = "cards_patch_229_434.json"
PATCH_TSV_FILE = "anki_patch_229_434.tsv"
PATCH_PROGRESS_FILE = "progress_patch_229_434.json"

BATCH_SIZE = 10
SLEEP_SECONDS = 1.1

START_BATCH = 409
END_BATCH = 434


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


def process_batch(entries_batch: list, batch_id: int):
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

        cards = [model_dump_card(card) for card in parsed.cards]
        return True, cards

    except Exception as e:
        error_path = f"error_batch_{batch_id}.json"
        save_json(error_path, {
            "batch_id": batch_id,
            "error": str(e),
            "entries_batch": entries_batch
        })
        print(f"批次 {batch_id} 出错，已写入 {error_path}")
        return False, []


def make_tags(card: dict) -> str:
    base = ["GermanLexicon", "patch_229_434"]
    if card["typ"] == "aktiv":
        base += ["aktiv", "productive"]
    elif card["typ"] == "passiv":
        base += ["passiv", "recognition"]
    else:
        base += ["kollokation", "collocation"]
    return " ".join(base)


def build_front_text(card: dict) -> str:
    if card["typ"] == "aktiv":
        return f"{card.get('zh', '')}"
    if card["typ"] == "passiv":
        return f"{card.get('wort', '')}"
    return f"{card.get('wort', '')}（固定搭配）"


def build_back_text(card: dict) -> str:
    lines = []
    lines.append(f"Wort: {card.get('wort', '')}")
    lines.append(f"Typ: {card.get('typ', '')}")

    if card.get("wortart"):
        lines.append(f"Wortart: {card.get('wortart')}")
    if card.get("genus"):
        lines.append(f"Genus: {card.get('genus')}")
    if card.get("zh"):
        lines.append(f"中文: {card.get('zh')}")
    if card.get("en"):
        lines.append(f"English: {card.get('en')}")
    if card.get("deutsch_def"):
        lines.append(f"Deutsch erklärt: {card.get('deutsch_def')}")
    if card.get("vergleich"):
        lines.append(f"辨析: {card.get('vergleich')}")
    if card.get("hinweis"):
        lines.append(f"说明: {card.get('hinweis')}")

    beispiele = card.get("beispiele", [])
    if beispiele:
        lines.append("")
        lines.append("例句:")
        for i, ex in enumerate(beispiele, 1):
            lines.append(f"{i}. {ex.get('satz', '')}")
            lines.append(f"   {ex.get('zh', '')}")

    kollokationen = card.get("kollokationen", [])
    if kollokationen:
        lines.append("")
        lines.append("搭配:")
        for k in kollokationen:
            lines.append(f"- {k}")

    items = card.get("items", [])
    if items:
        lines.append("")
        lines.append("固定搭配项:")
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item.get('搭配', '')}")
            lines.append(f"   {item.get('zh', '')}")
            lines.append(f"   {item.get('beispiel', '')}")

    return "<br>".join(esc(line) for line in lines)


def to_anki_row(card: dict) -> list[str]:
    front = build_front_text(card)
    back = build_back_text(card)
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


def main():
    entries = load_entries(INPUT_FILE)
    total_entries = len(entries)

    start_index = (START_BATCH - 1) * BATCH_SIZE
    end_index = min(END_BATCH * BATCH_SIZE, total_entries)

    patch_cards = load_json(PATCH_JSON_FILE, [])
    progress = load_json(PATCH_PROGRESS_FILE, {"done_batches": []})

    done_batches = set(progress.get("done_batches", []))

    print(f"补跑批次范围：{START_BATCH} ~ {END_BATCH}")
    print(f"对应词条索引范围：{start_index} ~ {end_index - 1}")
    print(f"已完成批次数：{len(done_batches)}")

    total_batches = END_BATCH - START_BATCH + 1

    for batch_id in range(START_BATCH, END_BATCH + 1):
        global_start = (batch_id - 1) * BATCH_SIZE
        if global_start >= total_entries:
            break

        if batch_id in done_batches:
            print(f"批次 {batch_id} 已完成，跳过。")
            continue

        batch = entries[global_start:global_start + BATCH_SIZE]
        print(f"处理批次 {batch_id} ({batch_id - START_BATCH + 1}/{total_batches}) ...")

        success, cards = process_batch(batch, batch_id)

        if success:
            for c in cards:
                c["_batch_id"] = batch_id
            patch_cards.extend(cards)
            done_batches.add(batch_id)

            save_json(PATCH_JSON_FILE, patch_cards)
            save_json(PATCH_PROGRESS_FILE, {
                "done_batches": sorted(done_batches),
                "start_batch": START_BATCH,
                "end_batch": END_BATCH
            })
            export_tsv(patch_cards, PATCH_TSV_FILE)
            print(f"批次 {batch_id} 成功，当前补跑卡片数：{len(patch_cards)}")
        else:
            print(f"批次 {batch_id} 失败，保留错误文件，稍后可重跑。")

        time.sleep(SLEEP_SECONDS)

    print()
    print("补跑完成。")
    print(f"JSON 文件：{PATCH_JSON_FILE}")
    print(f"TSV 文件：{PATCH_TSV_FILE}")
    print(f"进度文件：{PATCH_PROGRESS_FILE}")
    print(f"累计补跑卡片数：{len(patch_cards)}")
    print(f"累计成功批次数：{len(done_batches)} / {END_BATCH - START_BATCH + 1}")


if __name__ == "__main__":
    main()