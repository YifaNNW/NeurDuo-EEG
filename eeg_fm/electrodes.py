"""Mapping from EEG channel names to standard_1005 electrode coordinates and reference IDs."""

from __future__ import annotations

import functools

import mne
import numpy as np

OLD_TO_NEW = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

REFERENCE_TAG_IDS = {
    "": 0, "REF": 0,
    "LE": 1,
    "A1": 10, "A2": 11,
    "M1": 12, "M2": 13,
    "AVG": 14, "AV": 14, "AR": 14,
    "CZ": 15,
    "LER": 1,
}
DEFAULT_REFERENCE_ID = 0

_REF_SUFFIXES = ("REF", "LE", "LER", "A1", "A2", "M1", "M2", "AVG", "AV", "AR", "CZ")


def _unit(v) -> list[float]:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).tolist() if n > 0 else v.tolist()


@functools.lru_cache(maxsize=1)
def build_coord_dict() -> dict[str, list[float]]:
    montage = mne.channels.make_standard_montage("standard_1005")
    pos = montage.get_positions()["ch_pos"]
    d: dict[str, list[float]] = {}
    for name, xyz in pos.items():
        if xyz is None or np.any(np.isnan(xyz)):
            continue
        d[name.upper()] = _unit(xyz)
    for old, new in OLD_TO_NEW.items():
        if old not in d and new in d:
            d[old] = d[new]
        if new not in d and old in d:
            d[new] = d[old]
    return d


COORDS = build_coord_dict()


def _canonical_site(name: str) -> str | None:
    s = name.strip().upper()
    if not s:
        return None
    if s in COORDS:
        return s
    return None


def strip_reference(raw_name: str) -> tuple[str | None, str]:
    s = raw_name.strip().upper()
    s = s.replace("EEG ", "").replace("EEG", "").strip()
    ref_tag = ""
    if "-" in s:
        head, tail = s.split("-", 1)
        head = head.strip()
        tail = tail.strip()
        if tail in _REF_SUFFIXES:
            ref_tag = tail
            s = head
        else:
            s = head
    if s in ("A1", "A2", "M1", "M2", "REF", "AVG", "GND", "GROUND"):
        return None, ref_tag
    site = _canonical_site(s)
    return site, ref_tag


def reference_id(ref_tag: str) -> int:
    return REFERENCE_TAG_IDS.get(ref_tag.strip().upper(), DEFAULT_REFERENCE_ID)


def coord_of(site: str) -> list[float] | None:
    return COORDS.get(site.upper())


def montage_coords(montage_name: str) -> dict[str, list[float]]:
    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]
    out: dict[str, list[float]] = {}
    for name, xyz in pos.items():
        if xyz is None or np.any(np.isnan(xyz)):
            continue
        out[name.upper()] = _unit(xyz)
    return out


def map_channels(raw_names: list[str]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    used_sites: set[str] = set()
    for raw in raw_names:
        site, ref_tag = strip_reference(raw)
        if site is None or site in used_sites:
            continue
        used_sites.add(site)
        out[raw] = (site, ref_tag)
    return out
