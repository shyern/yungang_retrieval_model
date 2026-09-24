#!/usr/bin/env python3
"""Apply the merges/augmentations recorded in muti_positives.json to dataset.json.

Rules:
- Merged rows (entry_id is a list of 2 ids, or a comma-joined string):
  the entry whose name/description best matches the muti row is kept as the
  primary, gets the pooled image_path and the muti row's fields; the other
  entry is removed from dataset.json.
- Augmented rows (single id, image_path is a superset of the original):
  the entry's image_path/name/description/category are replaced by the muti
  row's values; the entries that previously owned the added images are
  removed from dataset.json (they were absorbed), unless they are themselves
  referenced by muti_positives.json (then images are shared).
- Name/description/category fixes are copied from muti; a corrupted
  non-string field in muti is skipped (original kept).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = ROOT / "data" / "dataset" / "dataset.json"
MUTI_PATH = ROOT / "data" / "dataset" / "muti_positives.json"


def images_of(record: dict) -> list[str]:
    value = record["image_path"]
    return value if isinstance(value, list) else [str(value)]


def norm(text: str) -> str:
    return "".join(text.split())


def resolve_ids(record: dict) -> list[str]:
    eid = record["entry_id"]
    if isinstance(eid, list):
        return [str(i) for i in eid]
    return [x.strip() for x in str(eid).split(",")]


def pick_primary(ids: list[str], muti_row: dict, full_by_id: dict) -> str:
    """Score each candidate id by field agreement with the muti row."""
    scores: list[tuple[int, str]] = []
    for i in ids:
        f = full_by_id[i]
        score = 0
        if f["name"] == muti_row["name"]:
            score += 2
        if norm(f["description"]) == norm(muti_row["description"]):
            score += 2
        elif norm(muti_row["description"]) and (
            norm(f["description"]).startswith(norm(muti_row["description"]))
            or norm(muti_row["description"]).startswith(norm(f["description"]))
        ):
            score += 1
        scores.append((score, i))
    return max(scores, key=lambda t: t[0])[1]  # ties keep the first listed id


def main() -> None:
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    muti = json.loads(MUTI_PATH.read_text(encoding="utf-8"))
    full_by_id = {d["entry_id"]: d for d in full}
    assert len(full_by_id) == len(full), "duplicate entry_id in dataset.json"

    # every id referenced by muti must exist in dataset.json
    muti_ids: set[str] = set()
    for m in muti:
        ids = resolve_ids(m)
        missing = [i for i in ids if i not in full_by_id]
        assert not missing, f"ids missing in dataset.json: {missing}"
        muti_ids.update(ids)

    merged_rows = [
        m for m in muti if isinstance(m["entry_id"], list) or "," in str(m["entry_id"])
    ]
    single_rows = [m for m in muti if m not in merged_rows]

    # merged rows: primary keeps its id, secondary is removed
    merge_updates: dict[str, dict] = {}
    removed_ids: set[str] = set()

    for m in merged_rows:
        ids = resolve_ids(m)
        primary = pick_primary(ids, m, full_by_id)
        secondaries = [i for i in ids if i != primary]
        assert secondaries, f"no secondary found for merged row {ids}"
        merge_updates[primary] = m
        removed_ids.update(secondaries)

    # single rows: augment image_path; absorbed owners (not referenced by
    # muti) are removed from dataset.json
    single_updates: dict[str, dict] = {}
    for m in single_rows:
        eid = m["entry_id"]
        f = full_by_id[eid]
        if images_of(m) != images_of(f):
            added = set(images_of(m)) - set(images_of(f))
            for image in added:
                owners = [
                    d["entry_id"]
                    for d in full
                    if d["entry_id"] != eid and image in images_of(d)
                ]
                assert len(owners) == 1, f"image {image} has {len(owners)} owners"
                owner = owners[0]
                if owner not in muti_ids:
                    removed_ids.add(owner)
        if any(m[k] != f[k] for k in ("image_path", "name", "description", "category")):
            single_updates[eid] = m

    # sanity: no overlap between removed ids and updated ids
    assert removed_ids.isdisjoint(merge_updates), removed_ids & set(merge_updates)
    assert removed_ids.isdisjoint(single_updates), removed_ids & set(single_updates)

    def apply_muti_row(record: dict, muti_row: dict) -> dict:
        out = dict(record)
        out["image_path"] = muti_row["image_path"]
        for key in ("name", "description", "category"):
            if isinstance(muti_row[key], str):
                out[key] = muti_row[key]
        return out

    new_full = []
    skipped_invalid_names = []
    for d in full:
        if d["entry_id"] in removed_ids:
            continue
        if d["entry_id"] in merge_updates:
            new_full.append(apply_muti_row(d, merge_updates[d["entry_id"]]))
        elif d["entry_id"] in single_updates:
            m = single_updates[d["entry_id"]]
            if not isinstance(m["name"], str):
                skipped_invalid_names.append(d["entry_id"])
            new_full.append(apply_muti_row(d, m))
        else:
            new_full.append(d)

    # ---- verification ----
    new_by_id = {d["entry_id"]: d for d in new_full}
    assert len(new_by_id) == len(new_full)
    assert len(new_full) == len(full) - len(removed_ids)

    # every muti row must be reflected in the new dataset
    for m in muti:
        ids = resolve_ids(m)
        if isinstance(m["entry_id"], list) or "," in str(m["entry_id"]):
            primary = [i for i in ids if i in new_by_id]
            assert len(primary) == 1, f"merged row lost: {ids}"
            assert images_of(new_by_id[primary[0]]) == images_of(m), ids
        else:
            assert new_by_id[ids[0]]["image_path"] == m["image_path"], ids[0]

    # image ownership: only the known shared pair may remain
    owners: dict[str, list[str]] = {}
    for d in new_full:
        for p in images_of(d):
            owners.setdefault(p, []).append(d["entry_id"])
    shared = {p: ids for p, ids in owners.items() if len(ids) > 1}
    # the only sanctioned sharing: images of v03_0157_014 (kept as its own
    # muti entry) that were also copied into v03_0157_013
    sanctioned = set(images_of(full_by_id["v03_0157_014"]))
    assert set(shared) <= sanctioned, f"unexpected shared images: {shared}"

    # no image may disappear except the one dropped by the merged v06 row
    old_images = {p for d in full for p in images_of(d)}
    new_images = {p for d in new_full for p in images_of(d)}
    dropped = old_images - new_images
    assert dropped <= {
        "images/06-第六卷-第8窟/v06_ref_p0074_2da6a3453493.jpg"
    }, f"unexpected dropped images: {dropped}"

    # ---- write ----
    backup = FULL_PATH.with_name("dataset.json.bak")
    shutil.copy2(FULL_PATH, backup)
    FULL_PATH.write_text(
        json.dumps(new_full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---- report ----
    print(f"merged rows:      {len(merged_rows)}")
    for m in merged_rows:
        ids = resolve_ids(m)
        primary = [i for i in ids if i in new_by_id][0]
        print(f"  {ids} -> kept {primary}, removed {[i for i in ids if i != primary]}")
    print(f"augmented rows:   {len([m for m in single_rows if images_of(m) != images_of(full_by_id[m['entry_id']])])}")
    print(f"removed entries:  {len(removed_ids)}")
    print(f"field fixes:      name={sum(1 for m in single_updates.values() if m['name'] != full_by_id[m['entry_id']]['name'])} "
          f"desc={sum(1 for m in single_updates.values() if m['description'] != full_by_id[m['entry_id']]['description'])}")
    if skipped_invalid_names:
        print(f"skipped corrupted name field in muti: {skipped_invalid_names}")
    if dropped:
        print(f"images dropped (per muti merge): {sorted(dropped)}")
    if shared:
        print(f"images now shared between entries: {shared}")
    print(f"entries: {len(full)} -> {len(new_full)}  (backup at {backup})")


if __name__ == "__main__":
    sys.exit(main())
