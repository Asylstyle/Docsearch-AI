"""Download the requested historical public PDFs; never bypass access/TLS errors."""
import argparse
import csv
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024


def safe_filename(name):
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,115}\.pdf", name)
            or ".." in name or name.split(".")[0].upper() in reserved):
        raise ValueError("unsafe PDF filename")
    return name


def encoded_url(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("expected public HTTPS URL without credentials")
    if re.search(r"%(?![0-9A-Fa-f]{2})", parts.path + parts.query):
        raise ValueError("invalid percent escape in URL")
    # Preserve existing %HH escapes while encoding Unicode and literal spaces once.
    return urlunsplit((parts.scheme, parts.netloc.encode("idna").decode("ascii"),
                      quote(parts.path, safe="/%:@!$&'()*+,;=-._~"),
                      quote(parts.query, safe="%=&/?@:!$'()*+,;~-._"), ""))


def pdf_info(path):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("missing, oversized or symlink PDF")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise ValueError("response is not PDF bytes (possibly an HTML error page)")
        source.seek(max(0, path.stat().st_size - 4096))
        if b"%%EOF" not in source.read():
            raise ValueError("PDF missing end marker; possibly partial download")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("encrypted PDF is not supported by ingestion")
    pages = len(reader.pages)
    if not pages:
        raise ValueError("PDF has no pages")
    text_pages, extraction_errors = 0, 0
    for page in reader.pages:
        try:
            text_pages += bool((page.extract_text() or "").strip())
        except Exception:
            extraction_errors += 1
    return pages, text_pages, extraction_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pdfs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ["filename", "url", "university", "topic"]:
            parser.error("CSV columns must be filename,url,university,topic")
        rows = list(reader)
    downloaded = skipped = errors = 0
    seen = set()
    for row in rows:
        temporary = None
        name = row.get("filename", "<missing>")
        try:
            safe_filename(name)
            if name in seen:
                raise ValueError("duplicate manifest filename")
            seen.add(name)
            target = args.output / name
            if target.is_symlink():
                raise ValueError("refusing symlink destination")
            if target.exists():
                try:
                    info = pdf_info(target)
                    print(f"SKIP {name} pages={info[0]} text_pages={info[1]} extraction_errors={info[2]}", flush=True)
                    skipped += 1
                    continue
                except Exception:
                    print(f"RETRY {name}: existing file is invalid; preserved until successful replacement", flush=True)
            request = Request(encoded_url(row["url"]), headers={"User-Agent": "DocSearch-AI-corpus-downloader/1.0"})
            deadline = time.monotonic() + 120
            with urlopen(request, timeout=30) as response:
                if response.status != 200 or urlsplit(response.url).scheme != "https":
                    raise ValueError("unexpected HTTP response or non-HTTPS redirect")
                if int(response.headers.get("Content-Length", "0")) > MAX_BYTES:
                    raise ValueError("PDF exceeds 10 MiB upload limit")
                with tempfile.NamedTemporaryFile(dir=args.output, prefix=name + ".", suffix=".part", delete=False) as destination:
                    temporary = Path(destination.name)
                    count = 0
                    while block := response.read(64 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError("download exceeded 120 seconds")
                        count += len(block)
                        if count > MAX_BYTES:
                            raise ValueError("download exceeds 10 MiB upload limit")
                        destination.write(block)
            info = pdf_info(temporary)
            temporary.replace(target)
            temporary = None
            downloaded += 1
            print(f"OK {name} bytes={count} pages={info[0]} text_pages={info[1]} extraction_errors={info[2]}", flush=True)
        except Exception as exc:
            errors += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"SUMMARY rows={len(rows)} downloaded={downloaded} skipped={skipped} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
