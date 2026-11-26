"""Core domain: types, configuration, imaging, model store, pipeline, jobs, CLI.

Never imports PySide6, torch or onnxruntime — the engines are injected into the pipeline, so
this package stays importable (and testable) on a machine with none of the extras installed.
"""
