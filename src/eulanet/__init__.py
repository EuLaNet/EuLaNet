"""
EuLaNet — Lean Eulerian-Lagrangian Flow Representation for ML / RL / Deep RL
Built by Sai Siddharth (Siddharthaerospace@gmail.com, saisidd@mit.edu)

Model-independent scientific data toolbox: FULL EULERIAN FLOW FIELD +
SPARSE LAGRANGIAN MATERIAL EVOLUTION + EXPLICIT SAME-TIME SPATIAL CORRESPONDENCE.
"""

__version__ = "0.1.0"
__author__ = "Sai Siddharth"
__email__ = "siddharthaerospace@gmail.com"

from . import eulerian, lagrangian, transport, correspondence, dataset, validation, io

__all__ = ["eulerian", "lagrangian", "transport", "correspondence", "dataset", "validation", "io"]
