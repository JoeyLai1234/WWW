"""Compatibility shim package for CLIP_dissect that re-exports the OpenAI CLIP API.
This allows code that imports `CLIP_dissect.clip` to use the standard `clip` package.
"""
from . import clip
