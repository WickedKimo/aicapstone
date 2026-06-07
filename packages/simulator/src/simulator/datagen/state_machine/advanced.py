"""State machine for the Franka advanced task: flip all shoes upright.

★ 只需改 _ACTIVE_SHOES 即可控制要翻的鞋子數量，其餘不用動。
  必須與 advanced_eval.py 的 ACTIVE_SHOES 保持一致。

每隻鞋經歷 8 個 phase：

  0. Hover above shoe    — 移到鞋子上方                 (gripper down, open)
  1. Approach            — 下降到抓取高度               (gripper down, open)
  2. Grasp               — 夾緊，保持高度               (gripper down, closed)
  3. Lift                — 提起至翻轉安全高度            (gripper down, closed)
  4. Flip-hold           — 保持位置，roll π→0 翻轉夾爪  (gripper up,   closed)
  5. Lower to place      — 下降到放置高度               (gripper up,   closed)
  6. Release             — 放開，鞋子落地朝上            (gripper up,   open)
  7. Retreat             — 退回高處，準備下一隻          (gripper open)

翻轉原理：phase 4 把 target roll 從 π（EE z 朝下）改為 0（EE z 朝上），
IK controller 在 _FLIP_HOLD 的步數內逐步完成 180° 旋轉，鞋子隨之翻正。

TUNING NOTES
------------
_GRASP_Z_OFFSET     夾爪太高抓空時降低；穿透鞋子時提高。
_LIFT_Z_OFFSET      需足夠高讓鞋子在翻轉時不碰桌面（> 鞋高 + 翻轉半徑）。
_PLACE_Z_OFFSET     gripper body 在放置時距鞋初始 z 的高度。
                    估算: finger_length + shoe_height，預設 0.15 m。
_GRASP_YAW_OFFSET   0 = 手指夾鞋的寬度方向；π/2 = 夾長度方向。
_GRIPPER_FLIP_ROLL_W  預設 0.0（roll π→0）。鞋落地方向不對時可試 math.pi*2。
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from leisaac.datagen.state_machine.base import StateMachineBase

# ---------------------------------------------------------------------------
# ★ 在這裡調整要啟用的鞋子（必須與 advanced_eval.py 的 ACTIVE_SHOES 相同）
# 可用名稱: "Sneaker" | "Blue_Sneaker" | "Worn_Rieker_Leather_Shoe"
# ---------------------------------------------------------------------------
_ACTIVE_SHOES: tuple[str, ...] = (
    "Sneaker",
    # "Blue_Sneaker",
    # "Worn_Rieker_Leather_Shoe",
)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
_EE_BODY_NAME = "panda_hand"
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

_GRIPPER_OPEN  = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.018
_MAX_ROT_DELTA       = 0.08
_IK_DLS_LAMBDA       = 0.01

_HOVER_Z_OFFSET:     float = 0.18
_GRASP_Z_OFFSET:     float = 0.06
_LIFT_Z_OFFSET:      float = 0.28
_FLIP_HOLD_Z_OFFSET: float = 0.25
_PLACE_Z_OFFSET:     float = 0.15   # 依鞋子尺寸調整
_RETREAT_Z_OFFSET:   float = 0.22

_GRIPPER_DOWN_ROLL_W:         float = math.pi
_GRIPPER_DOWN_PITCH_W:        float = 0.0
_GRIPPER_FLIP_ROLL_W:         float = 0.0   # roll π→0：夾爪翻轉 180°
_GRIPPER_DOWN_YAW_OFFSET_RANGE: tuple[float, float] = (-0.10, 0.10)
_GRASP_YAW_OFFSET:            float = 0.0   # 0 = 夾寬度方向；π/2 = 夾長度方向
_GRASP_RETREAT:               float = 0.01

_SUCCESS_MIN_UP_Y: float = 0.7   # local +Z 在 world frame 的 z 分量門檻

_FRANKA_REST_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4.0,
    "panda_joint3": 0.0,
    "panda_joint4": -3.0 * math.pi / 4.0,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2.0,
    "panda_joint7": math.pi / 4.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

# pick 順序直接從 _ACTIVE_SHOES 產生，不需額外改動
_PICK_ORDER: tuple[str, ...] = _ACTIVE_SHOES

# Phase 時間（步數）：0=hover, 1=approach, 2=grasp, 3=lift,
#                     4=flip-hold, 5=lower, 6=release, 7=retreat
_PHASE_DURATIONS_PER_SHOE: tuple[int, ...] = (150, 120, 20, 130, 160, 120, 15, 30)
_PHASES_PER_SHOE: int = len(_PHASE_DURATIONS_PER_SHOE)  # 8


# ---------------------------------------------------------------------------
# Module-level helpers（與 cutlery_arrangement.py 相同）
# ---------------------------------------------------------------------------

def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _retreat_xy_toward(
    target_pos_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    out = target_pos_w.clone()
    delta_xy = out[:, :2] - anchor_pos_w[:, :2]
    norm = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    out[:, :2] -= distance * (delta_xy / norm)
    return out


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    siny_cosp  = 2.0 * (w * z + x * y)
    cosy_cosp  = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])
    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)
    return -1


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AdvancedStateMachine(StateMachineBase):
    """Scripted Franka policy：依序將 _ACTIVE_SHOES 裡的每隻鞋翻正。

    The action vector is ``[panda_joint1, ..., panda_joint7, gripper]``.
    """

    MAX_STEPS: int = len(_ACTIVE_SHOES) * sum(_PHASE_DURATIONS_PER_SHOE) + 100

    def __init__(self) -> None:
        self._step_count:              int               = 0
        self._episode_done:            bool              = False
        self._ee_body_idx:             int               = -1
        self._jacobi_body_idx:         int               = -1
        self._arm_joint_ids:           list[int]         = []
        self._jacobi_joint_ids:        list[int]         = []
        self._rest_joint_pos:          torch.Tensor|None = None
        self._rest_ee_pos_w:           torch.Tensor|None = None
        self._initial_ee_pos_w:        torch.Tensor|None = None
        self._gripper_down_yaw_w:      torch.Tensor|None = None
        self._gripper_down_yaw_offset_w: torch.Tensor|None = None
        self._current_object_idx:      int               = 0
        self._event:                   int               = 0
        self._events_dt:               list[int]         = list(_PHASE_DURATIONS_PER_SHOE) * len(_PICK_ORDER)
        # hover phase 開始時記錄的鞋子位置，作為 phase 4-7 的放置基準
        self._shoe_place_pos_w:        torch.Tensor|None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)
        missing = [j for j in _FRANKA_ARM_JOINT_NAMES if j not in joint_names]
        if missing:
            raise ValueError(f"Missing Franka joints {missing} in {joint_names}")
        self._arm_joint_ids = [joint_names.index(j) for j in _FRANKA_ARM_JOINT_NAMES]

        if self._ee_body_idx < 0:
            raise ValueError(f"Could not find body '{_EE_BODY_NAME}' in Franka.")
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [jid + 6 for jid in self._arm_joint_ids]

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

    def check_success(self, env) -> bool:
        """所有 active 鞋子的 local +Z 在 world frame 的 z 分量 >= _SUCCESS_MIN_UP_Y。"""
        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for name in _ACTIVE_SHOES:
            q = env.scene[name].data.root_quat_w   # (N, 4) wxyz
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            up_y = 2.0 * (y * z + w * x)
            done = torch.logical_and(done, up_y >= _SUCCESS_MIN_UP_Y)
        return bool(done.all().item())

    def pre_step(self, env) -> None:
        pass

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device    = env.device
        num_envs  = env.num_envs
        obj_name  = _PICK_ORDER[self._current_object_idx]
        obj_pos_w  = env.scene[obj_name].data.root_pos_w.clone()
        obj_quat_w = env.scene[obj_name].data.root_quat_w.clone()
        robot_root_pos_w = robot.data.root_pos_w.clone()

        phase_in_cycle = self._event % _PHASES_PER_SHOE

        # ── 每隻鞋的 hover phase 開始時記錄初始 EE 位置與鞋子位置 ──────────
        if phase_in_cycle == 0 and self._step_count == 0:
            self._shoe_place_pos_w = obj_pos_w.clone()
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        place_target_w = (
            self._shoe_place_pos_w.clone()
            if self._shoe_place_pos_w is not None
            else obj_pos_w.clone()
        )

        # ── 夾爪朝向：phase 0-3 朝下；phase 4-7 翻轉（朝上）──────────────
        dtype = obj_quat_w.dtype
        if phase_in_cycle < 4:
            target_quat_w = self._gripper_down_quat_w(obj_quat_w, num_envs, device, dtype)
        else:
            target_quat_w = self._gripper_flipped_quat_w(obj_quat_w, num_envs, device, dtype)

        grasp_anchor_w = _retreat_xy_toward(obj_pos_w, robot_root_pos_w, _GRASP_RETREAT)

        # ── Phase 分派 ──────────────────────────────────────────────────────
        if phase_in_cycle == 0:
            target_pos_w, gripper_cmd = self._phase_hover(obj_pos_w, num_envs, device)
        elif phase_in_cycle == 1:
            target_pos_w, gripper_cmd = self._phase_approach(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 2:
            target_pos_w, gripper_cmd = self._phase_grasp(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 3:
            target_pos_w, gripper_cmd = self._phase_lift(obj_pos_w, num_envs, device)
        elif phase_in_cycle == 4:
            target_pos_w, gripper_cmd = self._phase_flip_hold(place_target_w, num_envs, device)
        elif phase_in_cycle == 5:
            target_pos_w, gripper_cmd = self._phase_lower(place_target_w, num_envs, device)
        elif phase_in_cycle == 6:
            target_pos_w, gripper_cmd = self._phase_release(place_target_w, num_envs, device)
        else:  # phase 7
            target_pos_w, gripper_cmd = self._phase_retreat(place_target_w, num_envs, device)

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_hover(self, obj_pos_w, num_envs, device):
        """從當前 EE 位置平滑插值到鞋子正上方。"""
        target = obj_pos_w.clone()
        target[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, obj_pos_w, num_envs, device):
        """保持 approach 高度夾緊，避免同時下壓與夾取導致空抓。"""
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, obj_pos_w, num_envs, device):
        """提起鞋子至安全高度（此時夾爪仍朝下）。"""
        target = obj_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_flip_hold(self, place_pos_w, num_envs, device):
        """保持位置不動，IK 逐步將 roll 從 π 旋轉到 0，完成 180° 翻轉。"""
        target = place_pos_w.clone()
        target[:, 2] += _FLIP_HOLD_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower(self, place_pos_w, num_envs, device):
        """翻轉後下降至放置高度（夾爪朝上，鞋底面向桌面）。"""
        target = place_pos_w.clone()
        target[:, 2] += _PLACE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_release(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _PLACE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_retreat(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _RETREAT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def advance(self) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        if self._step_count < self._events_dt[self._event]:
            return

        self._event += 1
        self._step_count = 0

        if self._event >= len(self._events_dt):
            self._episode_done = True
            return

        new_obj_idx = self._event // _PHASES_PER_SHOE
        if new_obj_idx != self._current_object_idx:
            # 切換到下一隻鞋：清除所有 per-shoe 快取
            self._current_object_idx    = new_obj_idx
            self._initial_ee_pos_w      = None
            self._gripper_down_yaw_w    = None
            self._gripper_down_yaw_offset_w = None
            self._shoe_place_pos_w      = None

    def reset(self) -> None:
        self._step_count            = 0
        self._episode_done          = False
        self._event                 = 0
        self._current_object_idx    = 0
        self._initial_ee_pos_w      = None
        self._gripper_down_yaw_w    = None
        self._gripper_down_yaw_offset_w = None
        self._shoe_place_pos_w      = None

    # ------------------------------------------------------------------
    # IK / control helpers（與 cutlery_arrangement.py 相同）
    # ------------------------------------------------------------------

    def _ee_pos_w(self, robot) -> torch.Tensor:
        idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_pos_w[:, idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_quat_w[:, idx, :]

    def _joint_position_franka_action(
        self,
        env,
        target_pos_w:  torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_cmd:   torch.Tensor,
    ) -> torch.Tensor:
        robot         = env.scene["robot"]
        root_pos_w    = robot.data.root_pos_w
        root_quat_w   = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root     = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root  = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w   = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w    = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root  = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos(
            pose_delta_root, self._ee_jacobian_root(robot)
        )
        joint_pos_target = self._clamp_arm_joint_pos(robot, joint_pos_target)
        return torch.cat([joint_pos_target, gripper_cmd], dim=-1)

    def _arm_joint_pos(self, robot) -> torch.Tensor:
        if not self._arm_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")
        return robot.data.joint_pos[:, self._arm_joint_ids]

    def _ee_jacobian_root(self, robot) -> torch.Tensor:
        if self._jacobi_body_idx < 0 or not self._jacobi_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")
        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ].clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_delta_joint_pos(self, pose_delta: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        jacobian_t    = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        delta_joint_pos = (
            jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)

    def _clamp_arm_joint_pos(self, robot, joint_pos: torch.Tensor) -> torch.Tensor:
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_joint_pos_limits[..., 0], arm_joint_pos_limits[..., 1])

    # ------------------------------------------------------------------
    # Gripper orientation helpers
    # ------------------------------------------------------------------

    def _gripper_down_quat_w(
        self,
        obj_quat_w: torch.Tensor,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """夾爪朝下（roll=π）。Yaw 在每隻鞋的第一次呼叫時快取，後續 phase 共用。"""
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = _yaw_from_quat_wxyz(obj_quat_w).to(device=device, dtype=dtype)
            self._gripper_down_yaw_offset_w = torch.empty(
                num_envs, device=device, dtype=dtype
            ).uniform_(*_GRIPPER_DOWN_YAW_OFFSET_RANGE)
            self._gripper_down_yaw_w = (
                base_yaw + _GRASP_YAW_OFFSET + self._gripper_down_yaw_offset_w
            ).clone()

        roll  = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W,  device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw   = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    def _gripper_flipped_quat_w(
        self,
        obj_quat_w: torch.Tensor,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """夾爪朝上（roll=0）。沿用 _gripper_down_quat_w 快取的 yaw，確保翻轉軸一致。

        roll π→0 讓 EE z 軸從朝下變朝上，鞋子跟著旋轉 180° 變回正面。
        """
        # 通常 _gripper_down_quat_w 已在 phase 0-3 初始化 yaw；這裡加個防呆
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = _yaw_from_quat_wxyz(obj_quat_w).to(device=device, dtype=dtype)
            self._gripper_down_yaw_offset_w = torch.zeros(num_envs, device=device, dtype=dtype)
            self._gripper_down_yaw_w = (base_yaw + _GRASP_YAW_OFFSET).clone()

        roll  = torch.full((num_envs,), _GRIPPER_FLIP_ROLL_W,  device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw   = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return _ACTIVE_SHOES