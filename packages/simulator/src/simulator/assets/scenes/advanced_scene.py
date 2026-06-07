from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from simulator import ASSETS_ROOT

"""Configuration for the Dining Room Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

ADVANCED_USD_PATH = str(SCENES_ROOT / "advanced" / "scene.usd")

ADVANCED_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ADVANCED_USD_PATH,
    )
)
