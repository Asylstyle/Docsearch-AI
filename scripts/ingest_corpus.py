"""Upload corpus PDFs, skipping exact filenames already ready in the API."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from download_corpus import pdf_info, safe_filename

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--directory", type=Path, default=ROOT / "data/pdfs")
    args = parser.parse_args()
    parts = urlsplit(args.api_url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        parser.error("expected API base URL without credentials/query/fragment")
    if not args.directory.is_dir():
        parser.error("PDF directory missing; run download_corpus.py first")
    # Abort before any upload if readiness cannot be established safely.
    try:
        with urlopen(args.api_url.rstrip("/") + "/documents", timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"unexpected HTTP {response.status}")
            documents = json.loads(response.read(10 * 1024 * 1024))
        if not isinstance(documents, list) or any(
            not isinstance(doc, dict) or not isinstance(doc.get("filename"), str)
            or not isinstance(doc.get("status"), str) or "pages_count" not in doc
            for doc in documents
        ):
            raise ValueError("invalid document listing")
        ready = {doc["filename"]: doc["pages_count"] for doc in documents if doc["status"] == "ready"}
    except Exception as exc:
        print(f"ERROR listing documents: {type(exc).__name__}; no uploads attempted", flush=True)
        return 1
    paths = sorted(args.directory.glob("*.pdf"))
    print("filename,status,pages_count", flush=True)
    succeeded = skipped = errors = 0
    for path in paths:
        try:
            safe_filename(path.name)
            if path.name in ready:
                print(f"{path.name},skipped_ready,{ready[path.name]}", flush=True)
                print("SKIP_REASON exact filename already has a ready document", flush=True)
                skipped += 1
                continue
            pdf_info(path)
            boundary = "docsearch-corpus-" + uuid.uuid4().hex
            body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: application/pdf\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
            request = Request(args.api_url.rstrip("/") + "/documents", method="POST", data=body,
                              headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
            with urlopen(request, timeout=300) as response:
                if response.status not in (200, 201):
                    raise ValueError(f"unexpected HTTP {response.status}")
                result = json.loads(response.read(1024 * 1024))
            if not isinstance(result, dict) or "id" not in result or "status" not in result or "pages_count" not in result:
                raise ValueError("API response missing document fields")
            print(f"{path.name},{result['status']},{result['pages_count']}", flush=True)
            print(f"DOCUMENT_ID filename={path.name} id={result['id']}", flush=True)
            if result["status"] == "ready":
                succeeded += 1
                ready[path.name] = result["pages_count"]
            else:
                errors += 1
        except HTTPError as exc:
            errors += 1
            # Do not print provider bodies, document contents or credentials.
            print(f"{path.name},ERROR_HTTP_{exc.code},unknown", flush=True)
        except Exception as exc:
            errors += 1
            print(f"{path.name},ERROR_{type(exc).__name__},unknown: {exc}", flush=True)
    print(f"SUMMARY files={len(paths)} uploaded={succeeded} skipped={skipped} errors={errors}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
