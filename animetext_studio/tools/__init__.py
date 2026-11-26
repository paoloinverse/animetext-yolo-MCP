"""Dev/demo entry points (run as `python -m animetext_studio.tools.<name>`).

These are thin, dependency-light harnesses around the engines so each part can be
verified from a shell without the GUI. They share the structured-error exit codes
from core.types (decision D8): 0 = success, 20 = not implemented, others per error.
"""
