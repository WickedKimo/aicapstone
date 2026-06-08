import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from simulator import ASSETS_ROOT
from simulator.utils.object_poses_loader import ObjectPoseConfig
from simulator.assets.scenes.advanced_scene import ADVANCED_CFG, ADVANCED_USD_PATH

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

ADVANCED_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "advanced" / "objects"

# ---------------------------------------------------------------------------
# ★ 在這裡調整要啟用的鞋子（必須與 advanced.py 的 _ACTIVE_SHOES 保持一致）
# 可用名稱: "Sneaker" | "Blue_Sneaker" | "Worn_Rieker_Leather_Shoe"
# ---------------------------------------------------------------------------
ACTIVE_SHOES: tuple[str, ...] = (
    "Sneaker",
    # "Blue_Sneaker",
    # "Worn_Rieker_Leather_Shoe",
)

# ---------------------------------------------------------------------------
# AprilTag → 物件名稱對應（只保留 ACTIVE_SHOES 的鞋子）
# ---------------------------------------------------------------------------
_ALL_TAG_TO_OBJECT: dict[int, str] = {
    2: "Sneaker",
    3: "Blue_Sneaker",
    4: "Worn_Rieker_Leather_Shoe",
}
TAG_TO_OBJECT: dict[int, str] = {
    tag: name for tag, name in _ALL_TAG_TO_OBJECT.items() if name in ACTIVE_SHOES
}

ANCHOR_TAG_ID: int = 0
# Anchor 貼在桌面固定位置，用來將 tag 座標系換算成 world 座標。
# 請依實際 AprilTag 貼紙位置調整。
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.40, 0.10, 0.0)

# 鞋子翻轉任務：起始時鞋子為「倒扣」狀態（繞 X 軸轉 π），
# object_pose_cfg 會將此姿態套用至 tag 偵測到的位置上。
OBJECT_Z: float = 0.05
OBJECT_ROLL: float = math.pi   # 倒扣：鞋面朝下、鞋底朝上
OBJECT_PITCH: float = 0.0

# 各鞋子 USD 的 yaw 修正值（rad）；使 spawn 後的視覺朝向與 gripper 座標系一致。
# 每個 USD 只需調整一次，有需要時再填。
PER_OBJECT_YAW_OFFSET: dict[str, float] = {
    "Sneaker":                  0.0,
    "Blue_Sneaker":             0.0,
    "Worn_Rieker_Leather_Shoe": 0.0,
}

# non-active 的鞋子若出現在 object_poses.json 中，直接忽略，不做 spawn。
IGNORED_OBJECT_NAMES: tuple[str, ...] = tuple(
    name for name in _ALL_TAG_TO_OBJECT.values() if name not in ACTIVE_SHOES
)


# ---------------------------------------------------------------------------
# 各鞋子的模擬起始中心位置與兩種側倒姿態
# ---------------------------------------------------------------------------
_SHOE_BASE_POS: dict[str, tuple[float, float, float]] = {
    "Sneaker":                  (0.35, -0.21, 0.1),
    "Blue_Sneaker":             (0.55, -0.10, 0.1),
    "Worn_Rieker_Leather_Shoe": (0.65, -0.10, 0.1),
}
 
_Q_LEFT_SIDEWAY: tuple[float, float, float, float]  = (0.5,  0.5,  0.5, -0.5)
_Q_RIGHT_SIDEWAY: tuple[float, float, float, float] = (0.5,  0.5, -0.5,  0.5)
 
 
# ---------------------------------------------------------------------------
# 自訂 reset event：每個 episode 隨機選左倒或右倒，位置加小 offset
# ---------------------------------------------------------------------------
def _randomize_shoe_sideways(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    center_pos: tuple[float, float, float],
    pos_range: dict[str, tuple[float, float]],
) -> None:
    """Reset shoe to left‑side or right‑side orientation with random position offset.
 
    local +Z 在側倒時指向水平方向（world ±X），world-Z 分量 ≈ 0 < 0.7，
    不會誤觸 success；只有正立（local +Z ↑）時才過關。
    """
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
 
    # ── 位置：中心 + 隨機 offset + env origin ──────────────────────────────
    pos = torch.tensor(list(center_pos), device=env.device, dtype=torch.float32)
    pos = pos.unsqueeze(0).expand(n, -1).clone()
    for dim, key in enumerate(("x", "y", "z")):
        if key in pos_range:
            lo, hi = pos_range[key]
            pos[:, dim] += torch.rand(n, device=env.device) * (hi - lo) + lo
    # 加上各環境的 origin offset（multi-env 相容）
    pos = pos + env.scene.env_origins[env_ids]
 
    # ── 姿態：隨機選 +90° 或 -90° 繞 Y ───────────────────────────────────
    q_left = torch.tensor(_Q_LEFT_SIDEWAY, device=env.device, dtype=torch.float32)
    q_right = torch.tensor(_Q_RIGHT_SIDEWAY, device=env.device, dtype=torch.float32)
    choose_left = (torch.rand(n, device=env.device) > 0.5).unsqueeze(1).expand(n, 4)
    rot = torch.where(choose_left,
                      q_left.unsqueeze(0).expand(n, -1),
                      q_right.unsqueeze(0).expand(n, -1)).contiguous()
 
    pose = torch.cat([pos, rot], dim=-1)  # (N, 7)  pos(3) + quat wxyz(4)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
 

# ---------------------------------------------------------------------------
# Scene（動態建構，只把 ACTIVE_SHOES 的 RigidObjectCfg 放進 scene config）
# ---------------------------------------------------------------------------
def _build_scene_cfg() -> type:
    attrs: dict = {
        "__annotations__": {"scene": AssetBaseCfg},
        "scene": ADVANCED_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene"),
    }
    for name in ACTIVE_SHOES:
        attrs[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ADVANCED_OBJECTS_ROOT / "Shoes" / f"{name}.usd"),
                mass_props=MassPropertiesCfg(mass=0.1),
            ),
            # 預設用「左倒」作為第一幀佔位；每次 reset 後由 event 覆蓋
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=_SHOE_BASE_POS[name],
                rot=_Q_LEFT_SIDEWAY,
            ),
        )
        attrs["__annotations__"][name] = RigidObjectCfg
    return configclass(type("AdvancedSceneCfg", (SingleArmFrankaTaskSceneCfg,), attrs))
 
 
AdvancedSceneCfg = _build_scene_cfg()


# ---------------------------------------------------------------------------
# 成功條件：所有 active 鞋子的 local +Z 在 world frame 的 Z 分量 >= min_up_z
# 公式：R[2,2] = 1 - 2*(x² + y²)，正立時 ≈ 1.0，側倒時 ≈ 0.0，倒扣時 ≈ -1.0
# ---------------------------------------------------------------------------
def shoes_upright(
    env,
    shoe_cfgs: list[SceneEntityCfg],
    min_up_z: float,
) -> torch.Tensor:
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in shoe_cfgs:
        obj: RigidObject = env.scene[cfg.name]
        q = obj.data.root_quat_w          # (N, 4) wxyz
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)   # world-Z component of local +Z
        done = torch.logical_and(done, up_z >= min_up_z)
    return done
 
 
@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    success = DoneTerm(
        func=shoes_upright,
        params={
            "shoe_cfgs": [SceneEntityCfg(name) for name in ACTIVE_SHOES],
            "min_up_z": 0.7,
        },
    )


# ---------------------------------------------------------------------------
# 環境設定
# ---------------------------------------------------------------------------

@configclass
class AdvancedEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the advanced shoe-flipping task environment."""

    scene: SingleArmFrankaTaskSceneCfg = AdvancedSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = "flip all shoes upright so the sole faces down."

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
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

        parse_usd_and_create_subassets(ADVANCED_USD_PATH, self)

        # 每個 episode reset 時：隨機選左倒／右倒，位置在中心點附近隨機擾動
        for shoe_name in ACTIVE_SHOES:
            setattr(
                self.events,
                f"reset_{shoe_name}_sideways",
                EventTermCfg(
                    func=_randomize_shoe_sideways,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(shoe_name),
                        "center_pos": _SHOE_BASE_POS[shoe_name],
                        "pos_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                    },
                ),
            )


        self.object_pose_cfg = ObjectPoseConfig(
            tag_to_object=TAG_TO_OBJECT,
            anchor_tag_id=ANCHOR_TAG_ID,
            anchor_world_pose=ANCHOR_WORLD_POSE,
            object_z=OBJECT_Z,
            object_roll=OBJECT_ROLL,
            object_pitch=OBJECT_PITCH,
            per_object_yaw_offset=PER_OBJECT_YAW_OFFSET,
            use_fixed_yaw=True,
            ignored_object_names=IGNORED_OBJECT_NAMES,
        )