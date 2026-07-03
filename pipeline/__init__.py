"""
__init__.py for the pipeline package.
Exposes the public API so dashboard/app.py can import cleanly.
"""
from pipeline.pipeline_core import inspect_cell, batch_inspect
from pipeline.database import init_db, save_result, load_history, get_summary
from pipeline.sensors  import read_weld_image, read_thermal, audio_to_spectrogram
from pipeline.models   import calculate_warranty_risk

__all__ = [
    "inspect_cell",
    "batch_inspect",
    "init_db",
    "save_result",
    "load_history",
    "get_summary",
    "read_weld_image",
    "read_thermal",
    "audio_to_spectrogram",
    "calculate_warranty_risk",
]
