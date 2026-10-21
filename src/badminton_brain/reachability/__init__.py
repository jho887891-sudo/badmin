# -*- coding: utf-8 -*-
"""P7-09 reachability + IK interface (simulation only).

Nothing here reads a camera, commands a robot or asserts a hit rate.  The module
answers "could the racket be at this pose, in time, with this evidence" and it
answers UNKNOWN when the evidence is missing.
"""
from .ik import (DampedLeastSquaresIk, IkNotConverged, IkNotImplemented, IkResult,
                 IkSolver, IkTarget, InvalidIkTarget, UnavailableIk)
from .kinematics import (DOF, FACE_FROM_LINK6_M, JOINT_LOWER_RAD, JOINT_NAMES,
                         JOINT_UPPER_RAD, LIMIT_SOURCE, MAX_FACE_REACH_M,
                         PROXY_SOURCE, face_position, forward_kinematics_face,
                         geometric_jacobian)
from .reachability import (Calibration, DetectionDimensionalityError, KinematicLimits,
                           ReachabilityDecision, ReachabilityReason, ReachabilityVerdict,
                           assess_reachability, require_world_3d, target_from_intercept)

__all__ = [
    "DOF", "FACE_FROM_LINK6_M", "JOINT_LOWER_RAD", "JOINT_NAMES", "JOINT_UPPER_RAD",
    "LIMIT_SOURCE", "MAX_FACE_REACH_M", "PROXY_SOURCE",
    "face_position", "forward_kinematics_face", "geometric_jacobian",
    "DampedLeastSquaresIk", "IkNotConverged", "IkNotImplemented", "IkResult", "IkSolver",
    "IkTarget", "InvalidIkTarget", "UnavailableIk",
    "Calibration", "DetectionDimensionalityError", "KinematicLimits",
    "ReachabilityDecision", "ReachabilityReason", "ReachabilityVerdict",
    "assess_reachability", "require_world_3d", "target_from_intercept",
]
