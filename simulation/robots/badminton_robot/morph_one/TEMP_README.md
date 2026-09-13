# Morph One TEMP parameters (must be replaced by measured values)

Spec: docs/simulation/BADMINTON_ROBOT.md S12 (TEMP policy) / S39 (TEMP -> REAL replacement).

| parameter | current value | status | source |
|---|---|---|---|
| chassis length | 0.70 m | `TEMP_PARAMETERIZED_PROXY` | engineering baseline chassis box |
| chassis width | 0.55 m | `TEMP_PARAMETERIZED_PROXY` | engineering baseline chassis box |
| wheel radius | 0.06 m | `TEMP_PARAMETERIZED_PROXY` | engineering baseline wheel radius |
| wheel width | 0.04 m | `TEMP_PARAMETERIZED_PROXY` | engineering baseline wheel width |
| wheel positions (robot_base) | `null` | `REQUIRES_MEASUREMENT` | wheel centers not measured |
| total mass / COM / inertia | `null` | `REQUIRES_MEASUREMENT` | base mass properties not measured |
| max steer angle | pi rad | `TEMP_PARAMETERIZED_PROXY` | continuous steering assumed |
| max steer rate | 6.0 rad/s | `TEMP_PARAMETERIZED_PROXY` | engineering baseline |
| max wheel speed | 40.0 rad/s | `TEMP_PARAMETERIZED_PROXY` | engineering baseline |

Frozen (NOT TEMP): the four-steer / four-drive topology and the eight semantic joints
(`fl_steer fl_drive fr_steer fr_drive rl_steer rl_drive rr_steer rr_drive`).
Per S63 it is forbidden to simplify this topology into a differential/mecanum/plain-4-wheel base.

Replacement procedure: fill the measured values, flip the status to
`VERIFIED_MEASURED`, run the robot tests, then the 1 -> 128 env regression (S72).
