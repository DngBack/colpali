"""
MLPReranker — non-graph baseline reranker for ablation.

This module re-exports the MLPReranker class from page_gat_reranker.py
to keep the models/ package clean and allow standalone imports.
"""

from .page_gat_reranker import MLPReranker  # noqa: F401
