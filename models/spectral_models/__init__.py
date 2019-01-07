"""
 Created by Narayan Schuetz at 07/12/2018
 University of Bern

 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


import importlib
from .registry import SPECTRAL_MODEL_REGISTRY


importlib.import_module("models.spectral_models.hybrid_block_models")
importlib.import_module("models.spectral_models.pure_block_models")
importlib.import_module("models.spectral_models.reference_models")

for m in SPECTRAL_MODEL_REGISTRY:
    globals()[m] = SPECTRAL_MODEL_REGISTRY[m]
