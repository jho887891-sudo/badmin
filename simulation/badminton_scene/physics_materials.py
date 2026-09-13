# -*- coding: utf-8 -*-
"""Physics materials - ENGINEERING_V0_1 initial values only (NOT identified).

All values here are engineering placeholders for stable contact; replace via
system identification with real hardware later. Single source of truth for
material parameters.
"""
COURT_MATERIAL = dict(name="court_material", friction=0.6, restitution=0.3, source="ENGINEERING_V0_1")
RACKET_MATERIAL = dict(name="racket_material", friction=0.4, restitution=0.55, source="ENGINEERING_V0_1")
NET_MATERIAL = dict(name="net_material", friction=0.5, restitution=0.3, source="ENGINEERING_V0_1")
SHUTTLE_MATERIAL = dict(name="shuttle_material", friction=0.5, restitution=0.45, source="ENGINEERING_V0_1")
ALL_MATERIALS = [COURT_MATERIAL, RACKET_MATERIAL, NET_MATERIAL, SHUTTLE_MATERIAL]
