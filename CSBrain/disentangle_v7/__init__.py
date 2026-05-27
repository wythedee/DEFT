"""Disentangle V7 (minimal): standard fine-tuning wrapper.

V7 is intentionally a thin wrapper around the original (non-disentangle) fine-tune
pipeline so we can incrementally add disentanglement losses later without forking
all dataset/model implementations.
"""

