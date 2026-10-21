# G6B_r06 · box_child_rigid

- gate_result: **PASS** (mode=landing)
- peak contact: 855.51348877 N, first contact step 12
- final chassis z: 0.33399990201 m, tail speed max: 0.000467113 m/s, tail z span: 0.0 m
- max tilt: 9.3463e-05 rad, tail max penetration: 9.8e-08 m
- contact api: isaaclab.sensors.ContactSensor(5 prims); per-wheel contact: MEASURED

## Positives
- gate all_finite = true
- gate contact_detected = true
- gate no_tip_over = true
- gate settles = true
- gate no_long_term_penetration = true
- gate per_wheel_contact_all_four = true

## Negatives / failing gates
- none

## Unknowns
- cooked PhysX geometry: UNKNOWN (no exposed query API in this build).
- determinism (same impl, two run_ids): PASS
