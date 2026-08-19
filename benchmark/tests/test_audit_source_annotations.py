import csv

from tools import audit_source_annotations as audit


semantic_token_sha256 = audit.semantic_token_sha256


def test_semantic_token_hash_normalizes_physical_line_endings_and_comments() -> None:
    head = (
        b'def f():\n'
        b'    """line one\nline two"""\n'
        b'    return 1\n'
    )
    annotated_windows_worktree = (
        b'# PAPER_DIFF[TEST-001][scope][low]: audit-only comment\r\n'
        b'def f():\r\n'
        b'    """line one\r\nline two"""\r\n'
        b'    return 1\r\n'
    )

    assert semantic_token_sha256(head) == semantic_token_sha256(
        annotated_windows_worktree
    )


def test_csv_annotation_update_is_idempotent(tmp_path, monkeypatch) -> None:
    csv_path = tmp_path / "source_differences.csv"
    fieldnames = ["diff_id", "annotation_status", "evidence"]
    rows = [
        {"diff_id": "VAE-001", "annotation_status": "old", "evidence": "keep-a"},
        {"diff_id": "FL-001", "annotation_status": "old", "evidence": "keep-b"},
        {"diff_id": "OTHER-001", "annotation_status": "untouched", "evidence": "keep-c"},
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(audit, "CSV_PATH", csv_path)

    first = audit.update_csv()
    first_bytes = csv_path.read_bytes()
    second = audit.update_csv()

    assert first["only_requested_cells_changed"] is True
    assert second["only_requested_cells_changed"] is True
    assert second["changed_cells"] == []
    assert csv_path.read_bytes() == first_bytes
