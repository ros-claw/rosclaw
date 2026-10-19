"""Bounded access metadata for original evidence, never a physical verdict.

Only successful host responses are recorded. Ranges describe serialized characters
actually returned, not complete JSON values or newly measured robot observations.
"""

import copy
import hashlib
import re
from collections import OrderedDict

from benchmarks.ros_expert_ablation.aci_context import encoded


class EvidenceReadProgress:
    def __init__(self):
        self.entries = OrderedDict()
        self.slices = OrderedDict()
        self.page_reads = 0
        self.repeated_pages = 0
        self.expand_reads = 0
        self.overlap_characters = 0
        self.file_catalog = None
        self.file_catalog_reads = 0
        self.file_reads = 0

    def observe(self, result, request):
        if request.get("operation") == "files.list" and result.get("scope") in {
            "IDENTICAL_ORIGINAL_FILE_ACCESS_NO_DIAGNOSIS_OR_AUTHORITY",
            "SAME_MODEL_GENERIC_ORIGINAL_FILES_NOT_CODEX_PRODUCT",
        }:
            self.file_catalog = copy.deepcopy(result)
            self.file_catalog_reads += 1
        if result.get("status") == "SEALED_ORIGINAL_EVIDENCE_NOT_ADMISSION":
            args = request.get("arguments", {})
            if request.get("operation") != "ros_engineer.read_task_evidence":
                return
            cursor = args.get("cursor", 0)
            identity = (result["common_observation_hash"], result["view"], cursor)
            self.page_reads += 1
            previous = self.entries.pop(identity, None)
            self.repeated_pages += int(previous is not None)
            self.entries[identity] = {
                "kind": "original_view_page",
                "common_observation_hash": result["common_observation_hash"],
                "view": result["view"],
                "cursor": cursor,
                "returned_records": len(result["records"]),
                "total_records": result["total_records"],
                "next_cursor": result["next_cursor"],
                "read_count": previous["read_count"] + 1 if previous else 1,
            }
        elif result.get("status") in {
            "ORIGINAL_SERIALIZED_RECORD_SLICE_NOT_NEW_OBSERVATION",
            "ORIGINAL_FILE_SLICE_NOT_DIAGNOSIS",
            "ORIGINAL_APPLICATION_INPUT_SLICE_NOT_DIAGNOSIS",
        }:
            is_file = result["status"] != "ORIGINAL_SERIALIZED_RECORD_SLICE_NOT_NEW_OBSERVATION"
            if is_file:
                args = request.get("arguments", {})
                application = result["status"] == "ORIGINAL_APPLICATION_INPUT_SLICE_NOT_DIAGNOSIS"
                operation = "ros_engineer.read_application_input" if application else "files.read"
                if request.get("operation") != operation or args.get("file") != result["file"]:
                    return
                ref = result["file"]
                catalog = (self.file_catalog or {}).get("files", [])
                metadata = next((row for row in catalog if row["file"] == ref), {})
                catalog_digest = metadata.get("sha256")
                response_digest = result.get("source_sha256") if application else None
                if application and (
                    not isinstance(response_digest, str)
                    or re.fullmatch("[0-9a-f]{64}", response_digest) is None
                ):
                    raise ValueError("original application input requires its host file digest")
                if response_digest and catalog_digest and response_digest != catalog_digest:
                    raise ValueError("application input differs from its original file catalog")
                digest = response_digest or catalog_digest
                hash_origin = (
                    "HOST_APPLICATION_INPUT_RESPONSE"
                    if response_digest
                    else "HOST_ORIGINAL_FILE_CATALOG"
                    if catalog_digest
                    else "UNAVAILABLE"
                )
                self.file_reads += 1
            else:
                ref = result["evidence_ref"]
                digest = result["record_sha256"]
            start = result["offset"]
            end = start + len(result["text"])
            total = result["total_characters"]
            # files.read permits an empty slice beyond EOF; it exposes no new range.
            if is_file and start > total and not result["text"]:
                start = end = total
            if not 0 <= start <= end <= total:
                raise ValueError("returned evidence slice outside declared original record")
            if (
                is_file
                and digest
                and start == 0
                and end == total
                and hashlib.sha256(result["text"].encode()).hexdigest() != digest
            ):
                raise ValueError("returned complete file differs from its original file digest")
            identity = ("file", ref, digest) if is_file else (ref, digest)
            previous = self.entries.pop(identity, None)
            ranges = previous["returned_character_ranges"] if previous else []
            overlap = sum(max(0, min(end, hi) - max(start, lo)) for lo, hi in ranges)
            self.overlap_characters += overlap
            self.expand_reads += int(not is_file)
            merged = []
            for lo, hi in sorted([*ranges, *([[start, end]] if end > start else [])]):
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            prefix = merged[0][1] if merged and merged[0][0] == 0 else 0
            self.entries[identity] = {
                "kind": "original_file_ranges" if is_file else "original_serialized_record_ranges",
                "evidence_ref": ref,
                "record_sha256": digest,
                "total_characters": total,
                "returned_character_ranges": merged,
                "next_unread_offset": prefix if prefix < total else None,
                "serialized_record_fully_returned": prefix == total,
                "read_count": previous["read_count"] + 1 if previous else 1,
            }
            slice_id = (*identity, start)
            self.slices.pop(slice_id, None)
            self.slices[slice_id] = {
                "evidence_ref": ref,
                "record_sha256": digest,
                "offset": start,
                "text": result["text"],
                "scope": (
                    "ORIGINAL_FILE_SLICE_NOT_CURRENT_STATE_OR_AUTHORITY"
                    if is_file
                    else "ORIGINAL_SERIALIZED_SLICE_NOT_COMPLETE_VALUE_OR_NEW_OBSERVATION"
                ),
            }
            if is_file:
                for row in [self.entries[identity], self.slices[slice_id]]:
                    row["file"] = row.pop("evidence_ref")
                    row["original_file_sha256"] = row.pop("record_sha256")
                    row["file_hash_origin"] = hash_origin
        # The episode itself has a hard turn cap; also bound this reusable helper.
        while len(self.entries) > 32:
            self.entries.popitem(last=False)
        while len(self.slices) > 16:
            self.slices.popitem(last=False)

    @staticmethod
    def _bounded(values, characters, *, utf8=False):
        selected = []
        for value in reversed(list(values)):
            text = encoded([value, *selected])
            if len(text.encode() if utf8 else text) <= characters:
                selected.insert(0, copy.deepcopy(value))
        return selected

    def payload(self):
        progress = self._bounded(self.entries.values(), 2000)
        records = [row for row in self.slices.values() if "file" not in row]
        slices = self._bounded(records, 2000)
        result = {
            "scope": "ORIGINAL_ACCESS_METADATA_NOT_LIVE_STATE_OR_AUTHORITY",
            "page_reads": self.page_reads,
            "repeated_page_reads": self.repeated_pages,
            "expand_reads": self.expand_reads,
            "overlapping_serialized_characters_returned": self.overlap_characters,
            "progress": progress,
            "progress_entries_not_inline": len(self.entries) - len(progress),
            "original_serialized_slices": slices,
            "slice_entries_not_inline": len(records) - len(slices),
        }
        if self.file_catalog_reads or self.file_reads:
            files = [row for row in self.slices.values() if "file" in row]
            visible = self._bounded(files, 12000, utf8=True)
            catalogs = self._bounded(
                [self.file_catalog] if self.file_catalog else [], 3500, utf8=True
            )
            result["original_file_access"] = {
                "scope": "RETURNED_ORIGINAL_FILES_NOT_DIAGNOSIS_L1_OR_AUTHORITY",
                "catalog_reads": self.file_catalog_reads,
                "slice_reads": self.file_reads,
                "catalog": catalogs[0] if catalogs else None,
                "catalog_omitted": self.file_catalog is not None and not catalogs,
                "file_slices": visible,
                "file_slices_not_inline": len(files) - len(visible),
                "file_slice_context_budget_utf8_bytes": 12000,
            }
        return result
