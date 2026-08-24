"""
Builds speaker-aware subtitle blocks from word-level items and
writes them out as an .srt file.
"""


def srt_time(seconds):
    ms = int((seconds % 1) * 1000)
    total = int(seconds)

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def clean_text(words):
    text = "".join(words)
    text = text.replace("  ", " ").strip()
    return text


def build_srt_blocks(word_items, max_line_duration=6.0, max_word_gap=0.8):
    blocks = []
    current = None

    for w in word_items:
        if current is None:
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "words": [w["word"]],
            }
            continue

        gap = w["start"] - current["end"]
        duration = current["end"] - current["start"]

        speaker_changed = w["speaker"] != current["speaker"]
        too_long = duration >= max_line_duration
        big_gap = gap >= max_word_gap

        if speaker_changed or too_long or big_gap:
            blocks.append(current)
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "words": [w["word"]],
            }
        else:
            current["end"] = w["end"]
            current["words"].append(w["word"])

    if current is not None:
        blocks.append(current)

    return blocks


def write_srt(blocks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, block in enumerate(blocks, start=1):
            text = clean_text(block["words"])

            f.write(f"{i}\n")
            f.write(f"{srt_time(block['start'])} --> {srt_time(block['end'])}\n")
            f.write(f"{block['speaker']}: {text}\n\n")

    print(f"Done. Saved: {output_path}")
