"""Render local gold PDF pages for human review; never evaluate retrieval or call LLM."""
import argparse
import html
import json
from pathlib import Path

from pypdf import PdfReader

from download_corpus import safe_filename

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "eval/questions.jsonl")
    parser.add_argument("--pdfs", type=Path, default=ROOT / "data/pdfs")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/review.md")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = ["# Ручная проверка вопросов", "",
              "Метки подготовлены ассистентом, человеком не проверены. Не считать их human-reviewed ground truth.",
              "Номера страниц — исходные PDF, с 1. Приведены только первые 1500 символов каждой gold-страницы.",
              "Цитата evidence может находиться дальше этого префикса; тогда нужен просмотр страницы целиком.",
              "Test зафиксирован: не менять вопросы/метки для настройки поиска. Retrieval-evaluation здесь нет.", ""]
    cache, errors = {}, 0
    for question in questions:
        output.extend([f"## {question['id']} · {question['type']} · {question['split']}", "",
                       html.escape(question["question"]), "",
                       f"answerable: {str(question['answerable']).lower()}", ""])
        if not question["answerable"]:
            output.extend(["**UNANSWERABLE — gold=[]; evidence пуст.**",
                           "Отсутствие ответа требует ручной проверки всего корпуса, а не одной страницы.", ""])
            continue
        output.extend(["Evidence:", "", "<pre>" + html.escape(question["evidence"]) + "</pre>", ""])
        for source in question["gold"]:
            try:
                filename = safe_filename(source["filename"])
                if filename not in cache:
                    path = args.pdfs / filename
                    if path.is_symlink():
                        raise ValueError("refusing symlink PDF")
                    reader = PdfReader(path)
                    cache[filename] = [(page.extract_text() or "") for page in reader.pages]
                pages = cache[filename]
                for page in source["pages"]:
                    if type(page) is not int or not 1 <= page <= len(pages):
                        raise ValueError("gold page outside PDF bounds")
                    output.extend([f"### {filename} · PDF-страница {page}", "",
                                   "<pre>" + html.escape(pages[page - 1][:1500]) + "</pre>", ""])
            except Exception as exc:
                errors += 1
                output.extend([f"**SOURCE ERROR: {type(exc).__name__}**", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    print(f"REVIEW questions={len(questions)} source_errors={errors} output={args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
