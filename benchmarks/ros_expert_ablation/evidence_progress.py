"""Bounded access metadata for original evidence, never a physical verdict.

Only successful host responses are recorded. Ranges describe serialized characters
actually returned, not complete JSON values or newly measured robot observations.
"""

import copy
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

    def observe(self, result, request):
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
        elif result.get("status") == "ORIGINAL_SERIALIZED_RECORD_SLICE_NOT_NEW_OBSERVATION":
            ref = result["evidence_ref"]
            digest = result["record_sha256"]
            start = result["offset"]
            end = start + len(result["text"])
            total = result["total_characters"]
            if not 0 <= start <= end <= total:
                raise ValueError("returned evidence slice outside declared original record")
            identity = (ref, digest)
            previous = self.entries.pop(identity, None)
            ranges = previous["returned_character_ranges"] if previous else []
            overlap = sum(max(0, min(end, hi) - max(start, lo)) for lo, hi in ranges)
            self.overlap_characters += overlap
            self.expand_reads += 1
            merged = []
            for lo, hi in sorted([*ranges, *([[start, end]] if end > start else [])]):
                if merged and lo <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            prefix = merged[0][1] if merged and merged[0][0] == 0 else 0
            self.entries[identity] = {
                "kind": "original_serialized_record_ranges",
                "evidence_ref": ref,
                "record_sha256": digest,
                "total_characters": total,
                "returned_character_ranges": merged,
                "next_unread_offset": prefix if prefix < total else None,
                "serialized_record_fully_returned": prefix == total,
                "read_count": previous["read_count"] + 1 if previous else 1,
            }
            self.slices.pop((ref, digest, start), None)
            self.slices[(ref, digest, start)] = {
                "evidence_ref": ref,
                "record_sha256": digest,
                "offset": start,
                "text": result["text"],
                "scope": "ORIGINAL_SERIALIZED_SLICE_NOT_COMPLETE_VALUE_OR_NEW_OBSERVATION",
            }
        # The episode itself has a hard turn cap; also bound this reusable helper.
        while len(self.entries) > 32:
            self.entries.popitem(last=False)
        while len(self.slices) > 16:
            self.slices.popitem(last=False)

    @staticmethod
    def _bounded(values, characters):
        selected = []
        for value in reversed(list(values)):
            if len(encoded([value, *selected])) <= characters:
                selected.insert(0, copy.deepcopy(value))
        return selected

    def payload(self):
        progress = self._bounded(self.entries.values(), 2000)
        slices = self._bounded(self.slices.values(), 2000)
        return {
            "scope": "ORIGINAL_ACCESS_METADATA_NOT_LIVE_STATE_OR_AUTHORITY",
            "page_reads": self.page_reads,
            "repeated_page_reads": self.repeated_pages,
            "expand_reads": self.expand_reads,
            "overlapping_serialized_characters_returned": self.overlap_characters,
            "progress": progress,
            "progress_entries_not_inline": len(self.entries) - len(progress),
            "original_serialized_slices": slices,
            "slice_entries_not_inline": len(self.slices) - len(slices),
        }
