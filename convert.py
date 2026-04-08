import json
import re
from pathlib import Path

INPUT_TXT = "dictionary.txt"
OUTPUT_JSON = "entries.json"


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def clean_line(line: str) -> str:
    line = line.replace("\x00", " ")
    line = re.sub(r"[\u0001-\u001f]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_noise_line(line: str) -> bool:
    if not line:
        return True

    # 单独页码
    if re.fullmatch(r"\d{1,4}", line):
        return True

    # 单个分栏字母，如 A / B / C
    if re.fullmatch(r"[A-ZÄÖÜ]", line):
        return True

    # 单个奇怪符号
    if re.fullmatch(r"[¸•·■◆]+", line):
        return True

    return False


def is_entry_start(line: str) -> bool:
    """
    识别新词条起点。
    适配：
    - ab|be|stel|len [...]
    - Ab|bruch [...]
    - Abend
    - Ab|druck [...]
    """

    if not line:
        return False

    # 最强特征：行首是德语词头，且包含 |
    if re.match(r"^[A-Za-zÄÖÜäöüß]+(?:\|[A-Za-zÄÖÜäöüß]+)+\b", line):
        return True

    # 兼容没有 | 的词头，但必须足够像词典词条
    # 例如：Abend / abseits
    # 后面通常接空格、音标、词性、逗号等
    if re.match(r"^[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{2,}(?:\s|\[|,|〈|$)", line):
        # 尽量排除普通解释句
        # 如果整行像自然句子，通常不会是词头
        words = line.split()
        if len(words) <= 4:
            return True

    return False


def extract_lemma(first_line: str) -> str:
    """
    从词头行提取 lemma。
    """
    first_line = clean_line(first_line)

    # 优先提取带 | 的词头
    m = re.match(r"^([A-Za-zÄÖÜäöüß]+(?:\|[A-Za-zÄÖÜäöüß]+)+)", first_line)
    if m:
        return m.group(1).replace("|", "")

    # 其次提取普通词头
    m = re.match(r"^([A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{2,})", first_line)
    if m:
        return m.group(1)

    return first_line.split()[0].strip(",.;:") if first_line.split() else "UNKNOWN"


def split_entries(text: str) -> list[dict]:
    raw_lines = text.split("\n")
    lines = [clean_line(line) for line in raw_lines]
    lines = [line for line in lines if not is_noise_line(line)]

    entries = []
    current = []

    for line in lines:
        if is_entry_start(line):
            if current:
                entries.append("\n".join(current).strip())
                current = []
        current.append(line)

    if current:
        entries.append("\n".join(current).strip())

    # 二次清洗：去掉过短、明显不是词条的块
    cleaned_entries = []
    for entry in entries:
        if len(entry) < 20:
            continue
        first_line = entry.split("\n", 1)[0].strip()
        lemma = extract_lemma(first_line)
        if lemma == "UNKNOWN":
            continue
        cleaned_entries.append(entry)

    result = []
    for i, entry in enumerate(cleaned_entries, start=1):
        first_line = entry.split("\n", 1)[0].strip()
        lemma = extract_lemma(first_line)
        result.append({
            "entry_id": f"{i:06d}",
            "lemma": lemma,
            "raw_text": entry
        })

    return result


def main():
    if not Path(INPUT_TXT).exists():
        print(f"未找到输入文件: {INPUT_TXT}")
        return

    text = load_txt(INPUT_TXT)
    entries = split_entries(text)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"切分完成，共生成 {len(entries)} 条词条 -> {OUTPUT_JSON}")

    # 预览前10条
    print("\n前10条预览：")
    for item in entries[:10]:
        preview = item['raw_text'].split("\n")[0]
        print(item["entry_id"], item["lemma"], "=>", preview)


if __name__ == "__main__":
    main()