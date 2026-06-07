import os
from isaaclab.app import AppLauncher

# 1. 啟動 Isaac Lab 核心環境
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 2. 環境啟動後，匯入 pxr 與專案資產路徑
from pxr import Usd, UsdPhysics
from simulator import ASSETS_ROOT

import gymnasium as gym
import isaaclab_tasks # 確保 Isaac Lab 的任務有被載入

# 印出所有包含 HCIS 的環境名稱
all_envs = list(gym.envs.registry.keys())
hcis_envs = [env for env in all_envs if "HCIS" in env]
print("目前註冊成功的 HCIS 環境有：", hcis_envs)

# 3. 關閉環境
simulation_app.close()