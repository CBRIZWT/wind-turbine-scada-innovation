"""Audit PAPER_DIFF annotations in the two mirrored author repositories.

This script is deliberately limited to two delivery artifacts:

* ``源码注释审计.json`` is regenerated from the current Git and AST evidence.
* The matching ``annotation_status`` cells in ``源码差异清单.csv`` are updated.

It never resets, cleans, stages, commits, installs dependencies, or runs models.
The Git ``HEAD`` blob is treated as the pre-annotation reference and the current
working-tree file as the post-annotation artifact.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tokenize
from datetime import datetime
from typing import Any


WORKSPACE = Path(r"E:\创新")
DELIVERY_DIR = WORKSPACE / "论文复现" / "复现基准_2026-07-19"
CSV_PATH = DELIVERY_DIR / "源码差异清单.csv"
JSON_PATH = DELIVERY_DIR / "源码注释审计.json"
ANNOTATION_STATUS = "comment-only AST-equivalent"

TARGETS = (
    {
        "diff_id": "VAE-001",
        "title": "Fault detection in wind turbines using health index monitoring with variational autoencoders",
        "repo": WORKSPACE
        / "论文复现"
        / "Fault detection in wind turbines using health index monitoring with variational autoencoders"
        / "wedowind-challenge-ASCE-EMI",
        "path": "Solution_ID8/src/model.py",
        "remote": "https://github.com/shun-wang1/wedowind-challenge-ASCE-EMI.git",
        "commit": "a14258007fe0ab083f921c355836c03b62a4ab81",
        "marker": "PAPER_DIFF[VAE-001][data_modality][high]",
    },
    {
        "diff_id": "FL-001",
        "title": "Wind turbine condition monitoring based on intra- and inter-farm federated learning",
        "repo": WORKSPACE
        / "论文复现"
        / "Wind turbine condition monitoring based on intra- and inter-farm federated learning"
        / "FL-Wind-NBM",
        "path": "federated_learning/FederatedAlgorithms.py",
        "remote": "https://github.com/EnergyWeatherAI/FL-Wind-NBM.git",
        "commit": "87230037f1556c225eecb3111153f4b6f1502268",
        "marker": "PAPER_DIFF[FL-001][benchmark_budget_and_data][high]",
    },
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def decode_python(data: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    return data.decode(encoding)


def ast_sha256(data: bytes) -> str:
    tree = ast.parse(decode_python(data))
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return sha256(canonical.encode("utf-8"))


def semantic_token_sha256(data: bytes) -> str:
    """Hash executable lexical tokens while excluding comments and layout trivia."""

    ignored = {
        tokenize.ENCODING,
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    tokens = [
        [
            token.type,
            token.string.replace("\r\n", "\n").replace("\r", "\n"),
        ]
        for token in tokenize.tokenize(io.BytesIO(data).readline)
        if token.type not in ignored
    ]
    payload = json.dumps(tokens, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(payload)


def git_status(repo: Path, *, expanded: bool) -> list[str]:
    args = ["status", "--porcelain=v1"]
    if expanded:
        args.append("--untracked-files=all")
    output = run_git(repo, *args).stdout.decode("utf-8", errors="strict")
    return [line for line in output.splitlines() if line]


def target_status_lines(lines: list[str], relative_path: str) -> list[str]:
    # Porcelain v1 starts with a two-character status and one space. Renames are
    # not expected for these files; matching the final path keeps the evidence
    # transparent without altering repository state.
    return [line for line in lines if line[3:] == relative_path]


def audit_target(spec: dict[str, Any]) -> dict[str, Any]:
    repo = Path(spec["repo"])
    relative_path = str(spec["path"])
    current_path = repo / Path(relative_path)
    if not repo.is_dir() or not current_path.is_file():
        raise FileNotFoundError(f"Missing repository or target: {current_path}")

    head = run_git(repo, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    remote = run_git(repo, "remote", "get-url", "origin").stdout.decode("utf-8").strip()
    base = run_git(repo, "show", f"{head}:{relative_path}").stdout
    current = current_path.read_bytes()

    base_ast_hash = ast_sha256(base)
    current_ast_hash = ast_sha256(current)
    base_token_hash = semantic_token_sha256(base)
    current_token_hash = semantic_token_sha256(current)
    marker = str(spec["marker"])
    base_text = decode_python(base)
    current_text = decode_python(current)
    compact_status = git_status(repo, expanded=False)
    expanded_status = git_status(repo, expanded=True)
    target_compact = target_status_lines(compact_status, relative_path)
    target_expanded = target_status_lines(expanded_status, relative_path)
    inferred_preexisting_compact = [
        line for line in compact_status if line not in target_compact
    ]
    inferred_preexisting = [line for line in expanded_status if line not in target_expanded]

    diff_check = run_git(repo, "diff", "--check", "--", relative_path, check=False)
    diff_text = run_git(repo, "diff", "--", relative_path).stdout.decode(
        "utf-8", errors="replace"
    )
    ast_equal = base_ast_hash == current_ast_hash
    token_equal = base_token_hash == current_token_hash
    marker_added_once = base_text.count(marker) == 0 and current_text.count(marker) == 1
    remote_matches = remote == spec["remote"]
    commit_matches = head == spec["commit"]
    target_modified = target_compact == [f" M {relative_path}"]
    verified = all(
        (
            ast_equal,
            token_equal,
            marker_added_once,
            remote_matches,
            commit_matches,
            target_modified,
            diff_check.returncode == 0,
        )
    )
    if not verified:
        raise RuntimeError(f"Annotation verification failed for {spec['diff_id']}")

    return {
        "diff_id": spec["diff_id"],
        "title": spec["title"],
        "repository_path": str(repo),
        "target_path": relative_path,
        "origin_remote": remote,
        "expected_remote": spec["remote"],
        "remote_matches_expected": remote_matches,
        "head_commit": head,
        "expected_commit": spec["commit"],
        "commit_matches_expected": commit_matches,
        "paper_diff_marker": marker,
        "marker_count_head": base_text.count(marker),
        "marker_count_worktree": current_text.count(marker),
        "marker_added_once": marker_added_once,
        "head_file_sha256": sha256(base),
        "worktree_file_sha256": sha256(current),
        "head_ast_sha256": base_ast_hash,
        "worktree_ast_sha256": current_ast_hash,
        "ast_equal": ast_equal,
        "head_semantic_token_sha256": base_token_hash,
        "worktree_semantic_token_sha256": current_token_hash,
        "semantic_tokens_equal": token_equal,
        "semantic_token_hash_contract": (
            "Comments, layout tokens, and physical CRLF/LF representation are excluded; "
            "Python executable lexical tokens remain."
        ),
        "byte_identical": base == current,
        "line_endings_head": {
            "crlf": base.count(b"\r\n"),
            "lf_only": base.count(b"\n") - base.count(b"\r\n"),
            "cr_only": base.count(b"\r") - base.count(b"\r\n"),
        },
        "line_endings_worktree": {
            "crlf": current.count(b"\r\n"),
            "lf_only": current.count(b"\n") - current.count(b"\r\n"),
            "cr_only": current.count(b"\r") - current.count(b"\r\n"),
        },
        "eof_newline_head": base.endswith((b"\n", b"\r")),
        "eof_newline_worktree": current.endswith((b"\n", b"\r")),
        "diff_check_exit_code": diff_check.returncode,
        "diff_check_stderr": diff_check.stderr.decode("utf-8", errors="replace").strip(),
        "target_status_compact": target_compact,
        "repository_status_compact": compact_status,
        "repository_status_expanded": expanded_status,
        "repository_dirty": bool(expanded_status),
        "preexisting_dirty_entries_compact_inferred_by_excluding_annotation_target": (
            inferred_preexisting_compact
        ),
        "preexisting_dirty_entry_count_compact_inferred": len(
            inferred_preexisting_compact
        ),
        "preexisting_dirty_entries_inferred_by_excluding_annotation_target": inferred_preexisting,
        "preexisting_dirty_entry_count_inferred": len(inferred_preexisting),
        "repository_state_was_not_cleaned_or_reset": True,
        "annotation_status": ANNOTATION_STATUS,
        "annotation_verification_passed": verified,
        "interpretation": (
            "AST and executable lexical tokens are identical to HEAD. The working-tree "
            "difference is non-executable annotation/line-ending trivia only; byte hashes "
            "differ. Any final-newline normalization is reported explicitly above."
        ),
        "unified_diff": diff_text,
    }


def atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def update_csv() -> dict[str, Any]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "diff_id" not in fieldnames or "annotation_status" not in fieldnames:
        raise ValueError("Unexpected source-difference CSV schema")

    original = [dict(row) for row in rows]
    target_ids = {str(spec["diff_id"]) for spec in TARGETS}
    matches = [row for row in rows if row.get("diff_id") in target_ids]
    if len(matches) != len(target_ids) or {row["diff_id"] for row in matches} != target_ids:
        raise ValueError("The CSV does not contain exactly one row for each audited diff_id")
    for row in matches:
        row["annotation_status"] = ANNOTATION_STATUS

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(CSV_PATH, buffer.getvalue(), encoding="utf-8-sig")

    changed_cells: list[dict[str, str]] = []
    for before, after in zip(original, rows, strict=True):
        for field in fieldnames:
            if before[field] != after[field]:
                changed_cells.append(
                    {
                        "diff_id": after["diff_id"],
                        "field": field,
                        "before": before[field],
                        "after": after[field],
                    }
                )
    expected_changes = {
        (diff_id, "annotation_status", ANNOTATION_STATUS) for diff_id in target_ids
    }
    actual_changes = {
        (cell["diff_id"], cell["field"], cell["after"]) for cell in changed_cells
    }
    if not actual_changes.issubset(expected_changes):
        raise RuntimeError(f"Unexpected CSV cell changes: {changed_cells}")
    all_target_cells_compliant = all(
        row["annotation_status"] == ANNOTATION_STATUS for row in matches
    )
    if not all_target_cells_compliant:
        raise RuntimeError("Audited CSV cells do not contain the required status")
    return {
        "path": str(CSV_PATH),
        "row_count": len(rows),
        "target_diff_ids": sorted(target_ids),
        "changed_cells": changed_cells,
        "only_requested_cells_changed": True,
        "all_target_cells_compliant": all_target_cells_compliant,
    }


def main() -> None:
    audits = [audit_target(spec) for spec in TARGETS]
    csv_update = update_csv()
    document = {
        "schema": "source_annotation_audit/v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "Two mirrored author repositories; comment/AST audit only; no training.",
        "reference_contract": "Git HEAD blob is pre-annotation; working tree is post-annotation.",
        "annotation_status": ANNOTATION_STATUS,
        "strict_json": True,
        "repositories_cleaned_or_reset": False,
        "training_run": False,
        "csv_update": csv_update,
        "audits": audits,
        "all_annotations_verified": all(
            audit["annotation_verification_passed"] for audit in audits
        ),
    }
    payload = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    atomic_write_text(JSON_PATH, payload + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "json_path": str(JSON_PATH),
                "csv_path": str(CSV_PATH),
                "audited_diff_ids": [audit["diff_id"] for audit in audits],
                "all_annotations_verified": document["all_annotations_verified"],
                "csv_only_requested_cells_changed": csv_update["only_requested_cells_changed"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
