# -*- coding: utf-8 -*-
"""t10_gui - open the Badminton Scene v0.1 in a real Isaac Sim GUI window."""
import argparse, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--steps", type=int, default=100000000)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = False          # GUI window on the X display
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from badminton_scene.scene_cfg import create_scene, SIM_DT

def main():
    sim, scene, robot, shuttle, _cs, _ex = create_scene(num_envs=args.num_envs)
    sim.set_camera_view([-6.5, -6.0, 3.2], [0.2, 0.0, 0.9])
    print('[T10] GUI scene ready - window should be visible on DISPLAY', flush=True)
    i = 0
    while simulation_app.is_running() and i < args.steps:
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
        i += 1
        if i % 2400 == 0:
            print('[T10] stepping, t=%.1fs' % (i * SIM_DT), flush=True)
    simulation_app.close()

if __name__ == '__main__':
    main()
