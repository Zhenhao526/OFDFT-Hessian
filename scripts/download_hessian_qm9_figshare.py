"""Download and audit the external Hessian QM9 Figshare dataset.

This dataset is not a drop-in replacement for the OFDFT ``QM9PBEForceFull``
labels because the current EG/EGF pipeline also needs density-derived labels.
The script keeps the external dataset under a separate directory and records the
Figshare manifest for later conversion or benchmark use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_ARTICLE_ID = 26363959
DEFAULT_OUTPUT_NAME = "HessianQM9Figshare"
DEFAULT_FALLBACK_VERSION = 4
USER_AGENT = "structures25-qm9-hessian-audit/1.0"


def _request(url: str, timeout: int) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _fetch_json(url: str, timeout: int) -> dict[str, Any]:
    with urllib.request.urlopen(_request(url, timeout), timeout=timeout) as response:
        return json.load(response)


def _download(url: str, path: Path, timeout: int, resume: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    existing = tmp_path.stat().st_size if resume and tmp_path.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    start = time.time()
    downloaded = existing
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response, tmp_path.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded and downloaded % (256 * 1024 * 1024) < len(chunk):
                    elapsed = max(time.time() - start, 1e-9)
                    print(
                        f"downloaded {downloaded / 1024**3:.2f} GiB "
                        f"to {tmp_path} at {downloaded / elapsed / 1024**2:.1f} MiB/s",
                        flush=True,
                    )
    except urllib.error.HTTPError as exc:
        if existing and exc.code == 416:
            tmp_path.replace(path)
            return
        raise
    tmp_path.replace(path)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _article_archive_url(article_id: int, version: int | None) -> str:
    if version is None:
        return f"https://figshare.com/ndownloader/articles/{article_id}"
    return f"https://figshare.com/ndownloader/articles/{article_id}/versions/{version}"


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)


def _find_hf_dataset_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in root.rglob("dataset_info.json"):
        if (path.parent / "state.json").exists() or list(path.parent.glob("*.arrow")):
            candidates.append(path.parent)
    return sorted(set(candidates))


def _audit_huggingface_dataset(root: Path, output_dir: Path) -> dict[str, Any]:
    candidates = _find_hf_dataset_dirs(root)
    audit: dict[str, Any] = {
        "root": root.as_posix(),
        "huggingface_dataset_dirs": [path.as_posix() for path in candidates],
        "datasets_importable": False,
        "datasets": [],
    }
    try:
        from datasets import load_from_disk  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency guard
        audit["datasets_import_error"] = repr(exc)
        return audit

    audit["datasets_importable"] = True
    for dataset_dir in candidates:
        row: dict[str, Any] = {"path": dataset_dir.as_posix()}
        try:
            ds = load_from_disk(dataset_dir.as_posix())
            row["type"] = type(ds).__name__
            if hasattr(ds, "keys"):
                row["splits"] = {}
                for split_name in ds.keys():
                    split = ds[split_name]
                    sample = split[0] if len(split) else {}
                    row["splits"][str(split_name)] = {
                        "num_rows": int(len(split)),
                        "columns": list(split.column_names),
                        "sample_summary": _sample_summary(sample),
                    }
            else:
                sample = ds[0] if len(ds) else {}
                row["num_rows"] = int(len(ds))
                row["columns"] = list(ds.column_names)
                row["sample_summary"] = _sample_summary(sample)
        except Exception as exc:
            row["error"] = repr(exc)
        audit["datasets"].append(row)
    _write_json(output_dir / "audit_huggingface_dataset.json", audit)
    return audit


def _shape_like(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    if isinstance(value, list):
        shape_list: list[int] = []
        current = value
        while isinstance(current, list):
            shape_list.append(len(current))
            current = current[0] if current else None
        return shape_list
    return None


def _sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in sample.items():
        shape = _shape_like(value)
        entry: dict[str, Any] = {"type": type(value).__name__}
        if shape is not None:
            entry["shape"] = shape
        elif isinstance(value, (int, float, str, bool)) or value is None:
            entry["value"] = value
        summary[key] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-id", type=int, default=DEFAULT_ARTICLE_ID)
    parser.add_argument("--article-version", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("DFT_DATA", "_runtime/external_datasets")) / DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-files", action="store_true")
    parser.add_argument("--download-archive", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    article_url = f"https://api.figshare.com/v2/articles/{args.article_id}"
    metadata: dict[str, Any] | None = None
    metadata_error: str | None = None
    try:
        metadata = _fetch_json(article_url, args.timeout)
        _write_json(output_dir / "figshare_article_metadata.json", metadata)
    except Exception as exc:
        metadata_error = repr(exc)
        print(f"WARNING: failed to fetch Figshare metadata: {metadata_error}", file=sys.stderr)

    version = args.article_version
    if version is None and metadata is not None:
        version_value = metadata.get("version")
        version = int(version_value) if version_value is not None else None
    if version is None:
        version = DEFAULT_FALLBACK_VERSION

    manifest: dict[str, Any] = {
        "article_id": args.article_id,
        "article_version": version,
        "article_api_url": article_url,
        "metadata_fetch_error": metadata_error,
        "files": [],
    }
    if metadata is not None:
        manifest.update(
            {
                "title": metadata.get("title"),
                "doi": metadata.get("doi"),
                "license": metadata.get("license"),
                "defined_type_name": metadata.get("defined_type_name"),
                "published_date": metadata.get("published_date"),
                "modified_date": metadata.get("modified_date"),
            }
        )
        for item in metadata.get("files", []):
            manifest["files"].append(
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "size": item.get("size"),
                    "download_url": item.get("download_url"),
                }
            )
    _write_json(output_dir / "figshare_manifest.json", manifest)

    if args.metadata_only or not (args.download_files or args.download_archive or args.extract or args.audit):
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.download_files:
        if metadata is None:
            raise SystemExit("Cannot download individual files without Figshare metadata.")
        for item in metadata.get("files", []):
            url = item.get("download_url")
            name = item.get("name")
            if not url or not name:
                continue
            _download(url, output_dir / "files" / str(name), args.timeout)

    archive_path = output_dir / f"hessian_qm9_figshare_article_{args.article_id}_v{version}.zip"
    if args.download_archive:
        _download(_article_archive_url(args.article_id, version), archive_path, args.timeout)

    extract_dir = output_dir / "extracted"
    if args.extract:
        if not archive_path.exists():
            raise SystemExit(f"Archive not found: {archive_path}")
        _extract_archive(archive_path, extract_dir)

    if args.audit:
        audit_root = extract_dir if extract_dir.exists() else output_dir
        audit = _audit_huggingface_dataset(audit_root, output_dir)
        print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
