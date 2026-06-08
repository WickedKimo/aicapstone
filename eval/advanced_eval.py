import math

import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.utils.seed import configure_seed

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from simulator import ASSETS_ROOT
from simulator.assets.scenes.advanced_scene import ADVANCED_CFG, ADVANCED_USD_PATH

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

ADVANCED_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "advanced" / "objects"

# ---------------------------------------------------------------------------
# ★ 在這裡調整要啟用的鞋子
# 可用名稱: "Sneaker" | "Blue_Sneaker" | "Worn_Rieker_Leather_Shoe"
# ---------------------------------------------------------------------------
ACTIVE_SHOES: tuple[str, ...] = (
    "_12xSneaker",
    # "Blue_Sneaker",
    # "Worn_Rieker_Leather_Shoe",
)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
TAG_TO_OBJECT: dict[int, str] = {2: "_12xSneaker", 3: "Blue_Sneaker", 4: "Worn_Rieker_Leather_Shoe"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.0, 0.0, 0.0)
OBJECT_Z: float = 0.12
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0
PER_OBJECT_YAW_OFFSET: dict[str, float] = {
    "_12xSneaker": 0.0,
    "Blue_Sneaker": 0.0,
    "Worn_Rieker_Leather_Shoe": 0.0,
}

# 各鞋子的起始中心位置（側倒）
_SHOE_ACTIVE_POS: dict[str, tuple[float, float, float]] = {
    "_12xSneaker":              (0.36, -0.21, 0.1),
    "Blue_Sneaker":             (0.55, -0.10, 0.1),
    "Worn_Rieker_Leather_Shoe": (0.65, -0.10, 0.1),
}
 
_S = math.sqrt(2.0) / 2.0
_Q_1: tuple[float, float, float, float] = (_S, 0.0, _S, 0.0)
_Q_2: tuple[float, float, float, float] = (0.0, _S, 0.0, _S)
_Q_3: tuple[float, float, float, float] = (_S, 0.0, -_S, 0.0)
_Q_4: tuple[float, float, float, float] = (0.0, _S, 0.0, -_S)
 
configure_seed(42)

# ---------------------------------------------------------------------------
# 自訂 reset event（與 env_cfg 共用相同邏輯）
# ---------------------------------------------------------------------------
def _randomize_shoe_sideways(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    center_pos: tuple[float, float, float],
    pos_range: dict[str, tuple[float, float]],
) -> None:
    """Reset shoe to left‑side or right‑side orientation with random position offset."""
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
 
    pos = torch.tensor(list(center_pos), device=env.device, dtype=torch.float32)
    pos = pos.unsqueeze(0).expand(n, -1).clone()
    for dim, key in enumerate(("x", "y", "z")):
        if key in pos_range:
            lo, hi = pos_range[key]
            pos[:, dim] += torch.rand(n, device=env.device) * (hi - lo) + lo
    pos = pos + env.scene.env_origins[env_ids]
 
    # 姿態：4 種隨機選 1
    qs = torch.tensor(
        [_Q_1, _Q_2, _Q_3, _Q_4],
        device=env.device, dtype=torch.float32,
    )  # (4, 4)
    idx = torch.randint(0, 4, (n,), device=env.device)
    rot = qs[idx]  # (N, 4)

    pose = torch.cat([pos, rot], dim=-1)
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Scene（動態建構，只把 ACTIVE_SHOES 的 RigidObjectCfg 放進 scene config，
#        inactive 的鞋子完全不進環境，不佔 physics 資源）
# ---------------------------------------------------------------------------

def _build_scene_cfg() -> type:
    """回傳一個只含 ACTIVE_SHOES 鞋子欄位的 AdvancedSceneCfg class。

    用 type() 動態建立 class 而非 @configclass 靜態宣告，
    確保 inactive 鞋子的 RigidObjectCfg 從未出現在 dataclass fields 裡，
    Isaac Lab scene manager 就不會 spawn 它們。
    """
    attrs: dict = {
        "__annotations__": {
            "scene": AssetBaseCfg,
        },
        # 場景根節點
        "scene": ADVANCED_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene"),
    }

    # 只加入 ACTIVE_SHOES 裡的鞋子，其餘完全不放入環境
    for name in ACTIVE_SHOES:
        attrs[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ADVANCED_OBJECTS_ROOT / "Shoes" / f"{name}.usd"),
                mass_props=MassPropertiesCfg(mass=0.15),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=_SHOE_ACTIVE_POS[name],
                rot=_Q_1,
            ),
        )
        attrs["__annotations__"][name] = RigidObjectCfg

    return configclass(
        type("AdvancedSceneCfg", (SingleArmFrankaTaskSceneCfg,), attrs)
    )


AdvancedSceneCfg = _build_scene_cfg()


# ---------------------------------------------------------------------------
# 成功條件：local +Z 的 world-Z 分量 >= min_up_z
# 正立 ≈ 1.0 ✓ │ 側倒 ≈ 0.0 ✗ │ 倒扣 ≈ -1.0 ✗
# ---------------------------------------------------------------------------
def shoes_upright(
    env,
    shoe_cfgs: list[SceneEntityCfg],
    min_up_z: float,
) -> torch.Tensor:
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for cfg in shoe_cfgs:
        obj: RigidObject = env.scene[cfg.name]
        q = obj.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)   # R[2,2] = world-Z of local +Z
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
# Env config
# ---------------------------------------------------------------------------

@configclass
class AdvancedEnvCfg(SingleArmFrankaTaskEnvCfg):
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

        # domain randomization 自動只對 active 鞋子生效
        for shoe_name in ACTIVE_SHOES:
            setattr(
                self.events,
                f"reset_{shoe_name}_sideways",
                EventTermCfg(
                    func=_randomize_shoe_sideways,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(shoe_name),
                        "center_pos": _SHOE_ACTIVE_POS[shoe_name],
                        "pos_range": {"x": (-0.1, 0.1), "y": (-0.03, 0.03)},
                    },
                ),
            )


TASK_ID = "Private-Advanced-Eval-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:AdvancedEnvCfg"},
)