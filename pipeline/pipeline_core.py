"""
pipeline_core.py
================
Core BattleEdge inspection pipeline — Steps 5-11.

Public API
----------
inspect_cell(cell_id, image_path, audio_path, thermal_path,
             voltage, current, weld_speed)
    → dict with all inspection fields

The function loads pre-trained models on first call (lazy singleton),
so repeated calls in a Streamlit session are fast.
"""

import os
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lazy-loaded model singletons — avoid reloading on every call
_anomaly_detector  = None
_defect_classifier = None


def _load_models():
    """Load or return cached model objects."""
    global _anomaly_detector, _defect_classifier

    if _anomaly_detector is None or _defect_classifier is None:
        from pipeline.models import AnomalyDetector, DefectClassifier
        models_dir = os.path.join(ROOT, "models")

        ad_path  = os.path.join(models_dir, "anomaly_detector.pkl")
        clf_path = os.path.join(models_dir, "defect_classifier.pkl")
        enc_path = os.path.join(models_dir, "label_encoder.pkl")

        if not os.path.exists(ad_path) or not os.path.exists(clf_path):
            raise FileNotFoundError(
                "Trained models not found.\n"
                "Run:  python pipeline/train.py"
            )

        _anomaly_detector  = AnomalyDetector.load(ad_path)
        _defect_classifier = DefectClassifier.load(clf_path, enc_path)

    return _anomaly_detector, _defect_classifier


# ─────────────────────────────────────────────────────────────────────────────
# Default simulated weld parameters per defect type
# Used when the dashboard doesn't supply real sensor readings
# ─────────────────────────────────────────────────────────────────────────────
_SIM_PARAMS = {
    "good_weld":      (22.0, 180.0, 0.50),
    "normal":         (22.0, 180.0, 0.50),
    "burn_through":   (30.0, 250.0, 0.30),
    "contamination":  (21.0, 175.0, 0.55),
    "lack_of_fusion": (18.0, 140.0, 0.70),
    "spatter":        (28.0, 230.0, 0.35),
    "porosity":       (23.0, 190.0, 0.45),
    "cold_weld":      (20.0, 155.0, 0.60),
    "misalignment":   (24.0, 200.0, 0.40),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function  — Step 11
# ─────────────────────────────────────────────────────────────────────────────

def inspect_cell(
    cell_id:      str,
    image_path:   str,
    audio_path:   str,
    thermal_path: str,
    voltage:      float = None,
    current:      float = None,
    weld_speed:   float = None,
    sim_defect:   str   = None,
) -> dict:
    """
    Run the full BattleEdge multi-modal inspection pipeline.

    Parameters
    ----------
    cell_id      : str   unique cell identifier, e.g. "CELL-001"
    image_path   : str   path to weld image (.jpg/.png)
    audio_path   : str   path to audio WAV file
    thermal_path : str   path to thermal PNG
    voltage      : float optional real process parameter (V)
    current      : float optional real process parameter (A)
    weld_speed   : float optional real process parameter (m/s)
    sim_defect   : str   hint used to look up simulated weld params
                         when voltage/current/weld_speed are not supplied

    Returns
    -------
    dict
        cell_id        : str
        anomaly_score  : float   (IsolationForest decision value)
        defect_type    : str
        defect_proba   : dict    {class: probability}
        thermal        : dict    {peak, mean, std}
        risk_score     : int     0-100
        decision       : "PASS" | "REJECT"
        latency_ms     : int
        image_path     : str
        audio_path     : str
        thermal_path   : str
    """
    t0 = time.time()

    # ── Step 1: Load sensors ─────────────────────────────────────────────────
    from pipeline.sensors import read_weld_image, read_thermal, audio_to_spectrogram

    img_array, _        = read_weld_image(image_path)
    spectrogram         = audio_to_spectrogram(audio_path)
    thermal_img, t_feat = read_thermal(thermal_path)

    # ── Step 2: Load models ──────────────────────────────────────────────────
    anomaly_det, defect_clf = _load_models()

    # ── Step 3: Anomaly score (acoustic) ────────────────────────────────────
    anomaly_score = anomaly_det.score(spectrogram)

    # ── Step 4: Defect type (weld parameters) ───────────────────────────────
    # Use supplied parameters; fall back to simulated defaults
    if voltage is None or current is None or weld_speed is None:
        hint = (sim_defect or "normal").lower()
        voltage, current, weld_speed = _SIM_PARAMS.get(hint, _SIM_PARAMS["normal"])

    defect_type  = defect_clf.predict(voltage, current, weld_speed)
    defect_proba = defect_clf.predict_proba(voltage, current, weld_speed)

    # ── Step 5: Warranty risk score ──────────────────────────────────────────
    from pipeline.models import calculate_warranty_risk
    risk_score = calculate_warranty_risk(anomaly_score, defect_type, t_feat)

    # ── Step 6: PASS / REJECT decision ──────────────────────────────────────
    decision = "REJECT" if risk_score >= 65 else "PASS"

    latency_ms = int((time.time() - t0) * 1000)

    return {
        "cell_id":       cell_id,
        "anomaly_score": round(anomaly_score, 4),
        "defect_type":   defect_type,
        "defect_proba":  defect_proba,
        "thermal":       t_feat,
        "risk_score":    risk_score,
        "decision":      decision,
        "latency_ms":    latency_ms,
        "image_path":    image_path,
        "audio_path":    audio_path,
        "thermal_path":  thermal_path,
        "weld_params":   {
            "voltage":    round(voltage,    2),
            "current":    round(current,    1),
            "weld_speed": round(weld_speed, 3),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch inspection
# ─────────────────────────────────────────────────────────────────────────────

def batch_inspect(cells: list, save_to_db: bool = True) -> list:
    """
    Run inspect_cell on a list of cell dicts and optionally save to DB.

    Parameters
    ----------
    cells : list of dicts, each with keys:
            cell_id, image_path, audio_path, thermal_path
            (optional: voltage, current, weld_speed, sim_defect)
    save_to_db : bool

    Returns
    -------
    list of result dicts
    """
    from pipeline.database import init_db, save_result
    if save_to_db:
        init_db()

    results = []
    for cell in cells:
        try:
            r = inspect_cell(**cell)
            if save_to_db:
                save_result(r)
            results.append(r)
            status = "✓" if r["decision"] == "PASS" else "✗"
            print(f"  {status}  {r['cell_id']:10s}  risk={r['risk_score']:3d}  {r['decision']}")
        except Exception as exc:
            print(f"  ✗  {cell.get('cell_id', '?')}  ERROR: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(ROOT, "data")

    test_cases = [
        {
            "cell_id":     "CELL-DEMO-001",
            "image_path":  os.path.join(DATA_DIR, "images",  "good_weld_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "normal_00.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "normal_0.png"),
            "sim_defect":  "good_weld",
        },
        {
            "cell_id":     "CELL-DEMO-002",
            "image_path":  os.path.join(DATA_DIR, "images",  "porosity_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "anomaly_00.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "porosity_0.png"),
            "sim_defect":  "porosity",
        },
        {
            "cell_id":     "CELL-DEMO-003",
            "image_path":  os.path.join(DATA_DIR, "images",  "burn_through_00.jpg"),
            "audio_path":  os.path.join(DATA_DIR, "audio",   "anomaly_03.wav"),
            "thermal_path": os.path.join(DATA_DIR, "thermal", "cold_weld_0.png"),
            "sim_defect":  "burn_through",
        },
    ]

    print("=" * 55)
    print("  BattleEdge — Pipeline Self-Test")
    print("=" * 55)
    results = batch_inspect(test_cases, save_to_db=True)
    for r in results:
        print(f"\n  {r['cell_id']}")
        print(f"    defect_type  : {r['defect_type']}")
        print(f"    anomaly_score: {r['anomaly_score']}")
        print(f"    thermal      : {r['thermal']}")
        print(f"    risk_score   : {r['risk_score']}/100")
        print(f"    decision     : {r['decision']}")
        print(f"    latency_ms   : {r['latency_ms']}")
