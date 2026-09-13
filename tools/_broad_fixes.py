from pathlib import Path

# --- D2 support: promote landed_within_horizon into the contract ---
tp = Path('/home/T7/ojh/robot_sim/src/badminton_brain/types.py')
s = tp.read_text()
if 'landed_within_horizon' not in s:
    old = '''    landing_point: Any = None
    arrival_time: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME'''
    new = '''    landing_point: Any = None
    arrival_time: Any = None
    # DEC-019: promoted from the predictor private attribute to the contract, because a
    # horizon-truncated landing point must never be consumed as a real bounce.
    landed_within_horizon: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME'''
    assert old in s, 'PredictedTrajectory header not found'
    s = s.replace(old, new)
    old_val = "        self.arrival_time = check_per_env_scalar('PredictedTrajectory.arrival_time', self.arrival_time)"
    new_val = old_val + chr(10) + "        if self.landed_within_horizon is not None:" + chr(10) + "            self.landed_within_horizon = check_per_env_scalar(" + chr(10) + "                'PredictedTrajectory.landed_within_horizon'," + chr(10) + "                np.asarray(self.landed_within_horizon, dtype=float))"
    assert old_val in s, 'arrival_time validation line not found'
    s = s.replace(old_val, new_val)
    tp.write_text(s)
    import ast; ast.parse(s)
    print('types.py: landed_within_horizon promoted (DEC-019)')
else:
    print('types.py already has landed_within_horizon')

# --- D5: TEMP racket offset must sit inside the TEMP safety workspace ---
ep = Path('/home/T7/ojh/robot_sim/src/badminton_brain/estimation/estimator.py')
e = ep.read_text()
if '(0.30, 0.0, 1.20)' not in e:
    old_off = '''RACKET_CONTACT_OFFSET = Param(
    (0.0, 0.0, 0.0), AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "identity placeholder until measured T_link6_tcp / T_tcp_contact are available",
)'''
    new_off = '''RACKET_CONTACT_OFFSET = Param(
    (0.30, 0.0, 1.20), AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP offset reproducing the nominal PiPER mount height until measured "
    "T_link6_tcp / T_tcp_contact exist; chosen to stay inside the TEMP safety workspace box",
)'''
    assert old_off in e, 'RACKET_CONTACT_OFFSET block not found'
    ep.write_text(e.replace(old_off, new_off))
    print('estimator.py: TEMP racket offset reconciled with the TEMP workspace (D5)')
else:
    print('estimator.py offset already reconciled')

# --- D6: final mode must see unresolved module parameters ---
vp = Path('/home/T7/ojh/robot_sim/src/badminton_brain/validation.py')
v = vp.read_text()
if 'measurement_requirements' not in v:
    anchor = "    for layer_name, param in config.stage_rate_hz.items():"
    block = '''    for module in registry.modules():
        collector = getattr(module, 'measurement_requirements', None)
        if callable(collector):
            try:
                requirements = collector() or {}
            except Exception as exc:  # a broken collector must not hide the problem
                errors.append(
                    f"layer '{module.layer.value}' measurement_requirements() raised {exc!r}")
                continue
            for name, param in sorted(requirements.items()):
                message = (f"layer '{module.layer.value}' parameter '{name}' is still "
                           f"{getattr(param, 'status', 'UNKNOWN')}")
                if mode == 'final':
                    errors.append('final mode requires measured parameters: ' + message)
                else:
                    warnings.append(message)
        unresolved = getattr(module, 'unresolved_limits', None)
        if callable(unresolved):
            try:
                items = unresolved() or ()
            except Exception as exc:
                errors.append(
                    f"layer '{module.layer.value}' unresolved_limits() raised {exc!r}")
                continue
            for item in items:
                message = f"layer '{module.layer.value}' has an unresolved limit {item!r}"
                if mode == 'final':
                    errors.append('final mode requires resolved limits: ' + message)
                else:
                    warnings.append(message)

'''
    assert anchor in v, 'validation anchor not found'
    v = v.replace(anchor, block + anchor)
    vp.write_text(v)
    import ast; ast.parse(v)
    print('validation.py: final mode now queries unresolved module parameters (D6)')
else:
    print('validation.py already queries module requirements')
