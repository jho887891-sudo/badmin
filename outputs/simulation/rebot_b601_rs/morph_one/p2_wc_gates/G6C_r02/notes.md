# G6C_r02 · mesh_convex_hull

- gate_result: **PASS** (mode=landing)
- peak contact: 2701.190917969 N, first contact step 12
- final chassis z: 0.333999663591 m, tail speed max: 0.000584411 m/s, tail z span: 0.0 m
- max tilt: 0.000187813 rad, tail max penetration: 3.36e-07 m
- contact api: isaaclab.sensors.ContactSensor(1 prims); per-wheel contact: UNKNOWN

## Positives
- gate all_finite = true
- gate contact_detected = true
- gate no_tip_over = true
- gate settles = true
- gate no_long_term_penetration = true

## Negatives / failing gates
- none

## Unknowns
- per-wheel contact is UNKNOWN for this implementation: only an explicit rigid body can host a ContactSensor, so A/C report an aggregate chassis value only (recorded as null, not 0 N).
- cooked PhysX geometry: UNKNOWN (no exposed query API in this build).
- determinism (same impl, two run_ids): PASS
