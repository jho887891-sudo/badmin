import hashlib, shutil, subprocess, sys, os

REPO = '/home/T7/ojh/robot_sim'
PY = './env_isaaclab/bin/python'
IMPL = os.path.join(REPO, 'src/badminton_brain/estimation/robot_localization.py')
TEST = os.path.join(REPO, 'tests/badminton_brain/test_robot_localization.py')

def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()

pristine = {IMPL: md5(IMPL), TEST: md5(TEST)}
for p in pristine:
    shutil.copy2(p, p + '.pristine')

MUTATIONS = [
    ('M1 Jacobian sign flip (F[:,0,2])', IMPL,
     'F[:, 0, 2] = -body_y * dt_col', 'F[:, 0, 2] = body_y * dt_col',
     ['CovarianceDynamicsTests']),
    ('M2 Q dropped entirely', IMPL,
     'Q = G @ u_cov @ np.swapaxes(G, 1, 2)',
     'Q = np.zeros_like(G @ u_cov @ np.swapaxes(G, 1, 2))',
     ['CovarianceDynamicsTests']),
    ('M3 odometry noise zeroed', IMPL,
     "sigma_v = self._noise['odom_translation_noise_m'] / dt_col",
     'sigma_v = np.zeros_like(dt_col)',
     ['CovarianceDynamicsTests']),
    ('M4 Joseph R-term (K R K^T) dropped', IMPL,
     'P_new = A @ P @ np.swapaxes(A, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)',
     'P_new = A @ P @ np.swapaxes(A, 1, 2)',
     ['CovarianceDynamicsTests']),
    ('M5 innovation wrap removed', IMPL,
     'innovation[:, 2] = wrap_angle(innovation[:, 2])',
     'innovation[:, 2] = innovation[:, 2]',
     ['CovarianceTests']),
    ('M6 orphaned (uncollected) acceptance test', TEST,
     '    def test_per_env_dt_and_one_dimensional_imu_arrays(self) -> None:',
     '    def _disabled_test_per_env_dt_and_one_dimensional_imu_arrays(self) -> None:',
     ['SuiteHygieneTests', 'BatchTests']),
    ('M7 yaw post-update wrap removed', IMPL,
     '    self._state[:, 2] = wrap_angle(self._state[:, 2])',
     '    self._state[:, 2] = self._state[:, 2]',
     ['CovarianceTests']),
]

print('pristine md5 impl=%s test=%s' % (pristine[IMPL][:12], pristine[TEST][:12]))
for name, path, old, new, tests in MUTATIONS:
    src = open(path + '.pristine', encoding='utf-8').read()
    if old not in src:
        print('%-42s PATCH TARGET NOT FOUND (skipped)' % name)
        continue
    open(path, 'w', encoding='utf-8').write(src.replace(old, new, 1))
    proc = subprocess.run([PY, 'tests/badminton_brain/test_robot_localization.py'] + tests,
                          cwd=REPO, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    verdict = 'RED (killed)' if 'FAILED' in out or 'ERROR' in out else 'GREEN (survived)'
    failed = [l.strip() for l in out.splitlines()
              if l.startswith('FAIL:') or l.startswith('ERROR:')]
    detail = [l.strip() for l in out.splitlines()
              if 'AssertionError' in l or 'Ran ' in l]
    print('%-42s %s' % (name, verdict))
    for line in failed[:2]:
        print('      ', line)
    for line in detail[-1:]:
        print('      ', line)
    for p in (IMPL, TEST):
        shutil.copy2(p + '.pristine', p)

ok = all(md5(p) == pristine[p] for p in pristine)
print('files restored byte-identical:', ok)
for p in pristine:
    os.remove(p + '.pristine')
