"""Parse official OhioT1DM XML into uniform 5-minute aligned series.

See Marling & Bunescu, "The OhioT1DM Dataset for Blood Glucose Level
Prediction: Update 2020" for field definitions. This module is for
research use only; do not redistribute raw XML.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

DT_5MIN = 300  # seconds

# 2020 BGLP challenge cohort (Empatica). Official protocol: exclude first hour of each
# test XML from official evaluation windows — we optionally drop first N steps.
CHALLENGE_2020_IDS = frozenset({540, 544, 552, 567, 584, 596})


def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts.strip(), "%d-%m-%Y %H:%M:%S")


def _events_child(root: ET.Element, tag: str) -> ET.Element:
    el = root.find(tag)
    if el is None:
        raise ValueError(f"Missing <{tag}> in patient XML")
    return el


def _glucose_series(root: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unix_seconds, values) for all CGM points, sorted."""
    g = _events_child(root, "glucose_level")
    times: List[float] = []
    vals: List[float] = []
    for ev in g.findall("event"):
        t = _parse_ts(ev.attrib["ts"]).timestamp()
        times.append(t)
        vals.append(float(ev.attrib["value"]))
    order = np.argsort(times)
    t_arr = np.asarray(times, dtype=np.float64)[order]
    v_arr = np.asarray(vals, dtype=np.float64)[order]
    return t_arr, v_arr


def _basal_step(root: ET.Element, grid: np.ndarray) -> np.ndarray:
    b = root.find("basal")
    if b is None:
        return np.zeros(len(grid), dtype=np.float64)
    events: List[Tuple[float, float]] = []
    for ev in b.findall("event"):
        events.append((_parse_ts(ev.attrib["ts"]).timestamp(), float(ev.attrib["value"])))
    events.sort(key=lambda x: x[0])
    if not events:
        return np.zeros(len(grid), dtype=np.float64)
    t_evt = np.array([e[0] for e in events], dtype=np.float64)
    v_evt = np.array([e[1] for e in events], dtype=np.float64)
    idx = np.searchsorted(t_evt, grid, side="right") - 1
    idx = np.clip(idx, 0, len(events) - 1)
    return v_evt[idx]


def _temp_basal_override(root: ET.Element, grid: np.ndarray, basal: np.ndarray) -> np.ndarray:
    out = basal.copy()
    tb = root.find("temp_basal")
    if tb is None:
        return out
    for ev in tb.findall("event"):
        t0 = _parse_ts(ev.attrib["ts_begin"]).timestamp()
        t1 = _parse_ts(ev.attrib["ts_end"]).timestamp()
        val = float(ev.attrib["value"])
        mask = (grid >= t0) & (grid < t1)
        out[mask] = val
    return out


def _bolus_insulin_per_bin(root: ET.Element, grid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid), dtype=np.float64)
    bol = root.find("bolus")
    if bol is None:
        return out
    for ev in bol.findall("event"):
        t0 = _parse_ts(ev.attrib["ts_begin"]).timestamp()
        t1 = _parse_ts(ev.attrib["ts_end"]).timestamp()
        dose = float(ev.attrib["dose"])
        # Spread dose across 5-minute bins touched by (t0, t1]
        idx = np.where((grid + DT_5MIN > t0) & (grid <= max(t1, t0 + 1.0)))[0]
        if len(idx) == 0:
            i = int(np.clip(np.searchsorted(grid, t0, side="right") - 1, 0, len(grid) - 1))
            idx = np.array([i])
        share = dose / max(len(idx), 1)
        out[idx] += share
    return out


def _meal_carbs_per_bin(root: ET.Element, grid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(grid), dtype=np.float64)
    m = root.find("meal")
    if m is None:
        return out
    for ev in m.findall("event"):
        if "carbs" not in ev.attrib:
            continue
        t = _parse_ts(ev.attrib["ts"]).timestamp()
        i = int(np.clip(np.searchsorted(grid, t, side="right") - 1, 0, len(grid) - 1))
        out[i] += float(ev.attrib["carbs"])
    return out


def _align_to_grid(t_gl: np.ndarray, v_gl: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """5-minute uniform grid with linear interpolation in mg/dL space."""
    if len(t_gl) < 2:
        raise ValueError("Need at least two CGM points")
    t0 = float(np.min(t_gl))
    t1 = float(np.max(t_gl))
    # Snap start to a 5-minute boundary for stability
    t0 = np.floor(t0 / DT_5MIN) * DT_5MIN
    grid = np.arange(t0, t1 + 0.1, DT_5MIN, dtype=np.float64)
    vals = np.interp(grid, t_gl, v_gl)
    return grid, vals.astype(np.float32)


def parse_patient_xml(xml_path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one patient XML file.

    Returns
    -------
    patient_id : int
    cgm : (T,) float32 mg/dL on 5-minute grid
    insulin : (T,) float32 — basal U/5min (approx) + bolus allocation per bin
    meals : (T,) float32 — carbs (g) reported in each 5-minute bin
    ehr : (47,) float32 — placeholder (OhioT1DM XML has no full EHR); zeros.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pid = int(root.attrib["id"])
    t_gl, v_gl = _glucose_series(root)
    grid, cgm = _align_to_grid(t_gl, v_gl)
    basal = _basal_step(root, grid)
    basal = _temp_basal_override(root, grid, basal)
    # U per 5 minutes ≈ basal_rate_U_per_h * (5/60)
    basal_u = basal.astype(np.float64) * (5.0 / 60.0)
    bolus_u = _bolus_insulin_per_bin(root, grid)
    insulin = (basal_u + bolus_u).astype(np.float32)
    meals = _meal_carbs_per_bin(root, grid).astype(np.float32)
    ehr = np.zeros(47, dtype=np.float32)
    return pid, cgm, insulin, meals, ehr


def discover_pairs(
    ohio_root: str,
    cohorts: Tuple[str, ...] = ("2018", "2020"),
) -> List[Tuple[str, str, int, str]]:
    """Return list of (train_path, test_path, patient_id, cohort_label)."""
    pairs: List[Tuple[str, str, int, str]] = []
    for cohort in cohorts:
        train_dir = os.path.join(ohio_root, cohort, "train")
        test_dir = os.path.join(ohio_root, cohort, "test")
        if not os.path.isdir(train_dir):
            continue
        for name in sorted(os.listdir(train_dir)):
            if not name.endswith("-ws-training.xml"):
                continue
            pid_str = name.split("-")[0]
            tid = f"{pid_str}-ws-testing.xml"
            train_p = os.path.join(train_dir, name)
            test_p = os.path.join(test_dir, tid)
            if not os.path.isfile(test_p):
                raise FileNotFoundError(f"Missing test XML for {name}: {test_p}")
            pairs.append((train_p, test_p, int(pid_str), cohort))
    return pairs


def load_train_test_patient(
    train_xml: str,
    test_xml: str,
    skip_test_head_steps: int = 0,
) -> Dict[str, object]:
    """Load official train/test XML; optionally drop first N test steps (2020 BGLP)."""
    pid_t, cgm_tr, ins_tr, meal_tr, ehr_tr = parse_patient_xml(train_xml)
    pid_e, cgm_te, ins_te, meal_te, ehr_e = parse_patient_xml(test_xml)
    if pid_t != pid_e:
        raise ValueError(f"Patient id mismatch {pid_t} vs {pid_e}")
    if skip_test_head_steps > 0:
        k = min(skip_test_head_steps, len(cgm_te) - 1)
        cgm_te = cgm_te[k:]
        ins_te = ins_te[k:]
        meal_te = meal_te[k:]
    # EHR: use train-file placeholder (identical structure); model may ignore.
    return {
        "patient_id": pid_t,
        "train": {
            "patient_id": pid_t,
            "cgm": cgm_tr,
            "insulin": ins_tr,
            "meals": meal_tr,
            "ehr": ehr_tr,
        },
        "test": {
            "patient_id": pid_t,
            "cgm": cgm_te,
            "insulin": ins_te,
            "meals": meal_te,
            "ehr": ehr_e,
        },
    }
