

def _roi(path, d):
    x0, x1 = d["world_x"]; y0, y1 = d["world_y"]; z0, z1 = d["z"]
    _box(path, (x1 - x0, y1 - y0, z1 - z0),
         ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0), d["color"])


def add_court_statics(prefix):
    c = COURT
    _box(prefix + "/Court/CourtFloor", (c["length"], c["doubles_width"], 0.004),
         (0.0, 0.0, 0.002), GROUND_VISUAL["color"])
    for (x, y, la, wa) in court_lines_geometry():
        nm = "L_X%.3f_Y%.3f" % (x, y)
        nm = nm.replace("-", "m").replace(".", "p")
        _box(prefix + "/Court/OuterLines/" + nm, (la, wa, 0.004), (x, y, LINE_HEIGHT_Z), LINE_COLOR)
    nb = net_visual_band_box()
    _box(prefix + "/Court/Net/NetVisual", (nb["size_x"], nb["size_y"], nb["size_z"]),
         (0.0, nb["y_center"], nb["z_center"]), (0.95, 0.95, 0.95))
    ncb = net_collision_box()
    _box(prefix + "/Court/Net/NetCollision", (ncb["size_x"], ncb["size_y"], ncb["size_z"]),
         (0.0, ncb["y_center"], ncb["z_center"]), (1.0, 0.9, 0.9), collision=True)
    posts = net_post_positions()
    _cyl(prefix + "/Court/Net/PostL", POST_RADIUS, NET["post_height"], posts[0], NET_VISUAL["post_color"])
    _cyl(prefix + "/Court/Net/PostR", POST_RADIUS, NET["post_height"], posts[1], NET_VISUAL["post_color"])
    _roi(prefix + "/Court/PerceptionROI", PERCEPTION_ROI)
    _roi(prefix + "/Court/CandidateStrikeROI", CANDIDATE_STRIKE_ROI)
    _roi(prefix + "/Court/SpawnRegion", SPAWN_REGION)
    _roi(prefix + "/Court/TargetZone", TARGET_ZONE)
    add_robot_visuals(prefix)


def add_robot_visuals(prefix):
    m = MORPH
    _box(prefix + "/Robot/MorphOnePlaceholder",
         (m["length_x"], m["width_y"], m["height_z"]), m["center_world"], (0.25, 0.35, 0.45))
    rig_x = CAMERA["rig_world"][0]; rig_z = CAMERA["rig_world"][2]
    _cyl(prefix + "/Robot/CameraRig/Mast", 0.02, rig_z, (rig_x, 0.0, rig_z / 2.0), (0.5, 0.5, 0.5))
    for nm, y in (("CameraLeftFrame", CAMERA["left_y"]), ("CameraRightFrame", -CAMERA["right_y"])):
        _box(prefix + "/Robot/CameraRig/" + nm, (0.05, 0.05, 0.05), (rig_x, y, rig_z), (0.95, 0.95, 0.25))
    _box(prefix + "/Robot/ComputeBoxPlaceholder", (0.20, 0.18, 0.08), (-1.60, 0.12, 0.29), (0.3, 0.3, 0.3))
    _box(prefix + "/Robot/BatteryPlaceholder", (0.30, 0.16, 0.08), (-1.60, -0.14, 0.29), (0.4, 0.4, 0.5))
    _box(prefix + "/Robot/EStopPlaceholder", (0.05, 0.05, 0.03), (-1.45, 0.22, 0.33), (1.0, 0.1, 0.1))


@configclass
class BadmintonSceneCfg(InteractiveSceneCfg):
    robot: object = ROBOT_ARTICULATION_CFG
    shuttle: object = SHUTTLE_CFG


def create_scene(num_envs: int = 1):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    scene = InteractiveScene(BadmintonSceneCfg(num_envs=num_envs,
                                               env_spacing=SIMULATION["env_spacing"]))
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/Ground", gp)
    light = sim_utils.DistantLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light.func("/World/Light", light, translation=(0.0, 0.0, 10.0))
    add_court_statics("/World/envs/env_0")
    sim.reset()
    robot = scene.articulations["robot"]
    shuttle = scene.rigid_objects["shuttle"]
    return sim, scene, robot, shuttle
