# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`PHONNIX_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`PHONNIX_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

PHOENIX_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/check/Documents/my_assets/phoenix3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "J1": 0.0,
            "J2": 0.0,
            "J3": 0.0,
            "J4": 0.0,
            "J5": 0.0,
            "J6": 0.0,
            "J7": 0.0,
            # "phoenix_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["J[1-4]"],
            effort_limit=70.0,
            velocity_limit=0.1,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["J[5-7]"],
            effort_limit=30.0,
            velocity_limit=0.1,
            stiffness=400.0,
            damping=80.0,
        )
        # "panda_hand": ImplicitActuatorCfg(
        #     joint_names_expr=["phoenix_finger_joint.*"],
        #     effort_limit=200.0,
        #     velocity_limit=0.2,
        #     stiffness=2e3,
        #     damping=1e2,
        # ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = PHOENIX_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
