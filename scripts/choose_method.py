"""Select once from dev; never read test metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['e5', 'hybrid_rrf', 'e5_rerank', 'hybrid_rerank']


def main():
    report = json.loads((ROOT/'eval/results/retrieval_dev.json').read_text(encoding='utf-8'))
    assert report['split'] == 'dev'
    n = report['answerable_count']
    eligible = [name for name in ORDER if report['methods'][name]['p50_ms'] <= 1500]
    if not eligible:
        raise RuntimeError('No method meets p50 <= 1500 ms; selection requires review')
    correct = {name: round(report['methods'][name]['Hit@1']*n) for name in eligible}
    best = max(correct.values())
    chosen = next(name for name in eligible if best-correct[name] <= 2)
    lines = ['# Выбор по dev', '', f'Выбран `{chosen}`. n={n}; dev run={report.get("run_id", report["created_at"])}.', '',
             'Сначала p50 <=1500 мс, затем максимум правильных top-1; среди методов',
             'в пределах двух правильных ответов от максимума выбран самый простой.',
             'Порядок сложности: ' + ' < '.join(ORDER) + '.', '',
             '| Метод | Hit@1 correct/n | p50 мс | Допущен |', '|---|---:|---:|---|']
    for name in ORDER:
        metric = report['methods'][name]
        lines.append(f'| {name} | {round(metric["Hit@1"]*n)}/{n} | {metric["p50_ms"]:.3f} | {name in eligible} |')
    lines += ['', 'Test при выборе не использован. Разметка подготовлена ассистентом, не человеком.', '']
    (ROOT/'eval/results/method_choice.md').write_text('\n'.join(lines), encoding='utf-8')
    print(chosen)


if __name__ == '__main__':
    main()
