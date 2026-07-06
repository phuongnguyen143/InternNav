import numpy as np
from internutopia.core.config.robot import RobotCfg
from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.robots.h1 import H1Robot


def _ensure_physx_articulation_attrs(articulation: IsaacsimArticulation) -> None:
    """Ensure custom USD assets define Physx articulation attributes before use."""
    from pxr import PhysxSchema
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(articulation._articulation.prim_path)
    art_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    if not art_api.GetEnabledSelfCollisionsAttr():
        art_api.CreateEnabledSelfCollisionsAttr().Set(True)
    if not art_api.GetSolverPositionIterationCountAttr():
        art_api.CreateSolverPositionIterationCountAttr()
    if not art_api.GetSolverVelocityIterationCountAttr():
        art_api.CreateSolverVelocityIterationCountAttr()


def _set_enabled_self_collisions_safe(self_art: IsaacsimArticulation, flag: bool) -> None:
    """Apply PhysxArticulationAPI when the USD omits enabledSelfCollisions metadata."""
    try:
        self_art._articulation.set_enabled_self_collisions(flag=flag)
    except Exception as e:
        err = str(e)
        if 'Empty typeName' not in err and 'enabledSelfCollisions' not in err:
            raise
        _ensure_physx_articulation_attrs(self_art)
        self_art._articulation.set_enabled_self_collisions(flag=flag)


@BaseRobot.register('VLNH1Robot')
class VLNH1Robot(H1Robot):
    def __init__(self, config: RobotCfg, scene: IScene):
        original = IsaacsimArticulation.set_enabled_self_collisions
        IsaacsimArticulation.set_enabled_self_collisions = _set_enabled_self_collisions_safe
        try:
            super().__init__(config, scene)
        finally:
            IsaacsimArticulation.set_enabled_self_collisions = original
        _ensure_physx_articulation_attrs(self.articulation)
        self.current_action = None

    def post_reset(self):
        super().post_reset()
        self._torso_link = self._rigid_body_map[self.config.prim_path + '/torso_link']
        self._imu_link = self._rigid_body_map[self.config.prim_path + '/imu_link']

    def apply_action(self, action: dict):
        import omni.isaac.core.utils.numpy.rotations as rot_utils

        self.current_action = action
        ret = super().apply_action(action)
        if 'topdown_camera_500' in self.sensors:
            orientation_quat = np.array([-0.70710678, 0.0, 0.0, 0.70710678])
            robot_pos = self.articulation.get_world_pose()[0]
            self.sensors['topdown_camera_500'].set_world_pose(
                [robot_pos[0], robot_pos[1], robot_pos[2] + 0.75],
                orientation_quat,
            )

        if 'topdown_camera_50' in self.sensors:
            orientation_quat = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            robot_pos = self.articulation.get_world_pose()[0]
            self.sensors['topdown_camera_50']._camera.set_pose(
                [robot_pos[0], robot_pos[1], robot_pos[2] + 0.75],
                orientation_quat,
            )

        return ret
