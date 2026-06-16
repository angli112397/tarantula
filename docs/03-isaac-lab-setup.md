# Isaac Sim / Isaac Lab 环境搭建排障记录（RTX 5060 Ti）

记录本机（RTX 5060 Ti / Blackwell sm_120，16GB VRAM，15GB 内存）从零搭建
Isaac Sim 4.5.0.0 + Isaac Lab 2.3.2 直到 M4 冒烟测试通过的全部坑，供后续
重装/换机/排障时参考。所有问题均已解决，当前环境可正常跑 IsaacLab
`AppLauncher` 脚本。

## 0. 最终基线

| 项目 | 值 |
|---|---|
| GPU | RTX 5060 Ti（Blackwell, sm_120 / CC 12.0），16GB VRAM |
| NVIDIA 驱动 | **580.159.03**（`nvidia-driver-580-open`，从 595.71.05 降级） |
| 系统内存 / swap | 15GB / **15GB**（`/swapfile` 2GB + `/swapfile2` 14GB，`/etc/fstab` 持久化） |
| Isaac Sim | 4.5.0.0，`pip install isaacsim[all,extscache]`，装在 `~/isaac_venv`（Python 3.10.12） |
| Isaac Lab | 2.3.2，`~/IsaacLab`，`./isaaclab.sh --install` 全量 editable 安装 |
| torch | 2.7.0+cu128 / torchvision 0.22.0+cu128 |
| 会话 | X11（`XDG_SESSION_TYPE=x11`），非 Wayland |

## 1. Blackwell + 驱动 595 → Vulkan 段错误 → 降级到 580.159.03

**现象**：`SimulationApp` 初始化时在 `_wait_for_viewport` 附近 Vulkan 段错误
崩溃（595.71.05 对 Blackwell sm_120 的 Vulkan 支持有问题）。

**修复**：`apt` 降级到 `nvidia-driver-580-open`（580.159.03），重启后
Vulkan 初始化正常。

**注意**：本机此前为解决 GPU 自动关机问题，在
`/media/ang/Kshatriya/Linux/5060ti-自动关机排查记录.md` 中记录了
"强制 X11 会话"修复（`/etc/gdm3/custom.conf` 设 `WaylandEnable=false` +
GRUB `nvidia-drm.modeset=1`）。驱动降级 + 重启后**该修复依然有效**
（已验证 `XDG_SESSION_TYPE=x11`、GRUB/GDM 配置未被驱动包覆盖）——
如果以后再换驱动版本，务必重启后复查这两个配置项。

## 2. 内存不足 → 加 swap

Isaac Sim 单进程峰值 RSS 可达 ~5-12GB，15GB 物理内存裕度太小，容易在
GPU 已经处于不稳定历史的机器上触发 OOM/自动关机。加了 15GB swap：

```bash
sudo fallocate -l 2G /swapfile   && sudo chmod 600 /swapfile   && sudo mkswap /swapfile   && sudo swapon /swapfile
sudo fallocate -l 14G /swapfile2 && sudo chmod 600 /swapfile2 && sudo mkswap /swapfile2 && sudo swapon /swapfile2
# 并把两行加入 /etc/fstab 持久化
```

## 3. `_wait_for_viewport` 无限循环挂起 + 内存泄漏 → `create_new_stage: False`

**现象**：裸 `SimulationApp({"headless": True, ...})`（不经过 IsaacLab
`AppLauncher`）初始化后卡死在 `_wait_for_viewport` 的 while 循环里不返回，
RSS 持续缓慢增长（内存泄漏）。

**修复**：`SimulationApp` 配置里加 `"create_new_stage": False`：

```python
simulation_app = SimulationApp({
    "headless": True,
    "anti_aliasing": 0,
    "denoiser": False,
    "create_new_stage": False,
})
```

之后手动 `omni.usd.get_context().new_stage()` 即可正常用。

**重要**：IsaacLab 的 `AppLauncher._resolve_viewport_settings()`
**默认就会设置 `create_new_stage=False`**（官方注释：
"avoid creating new stage at startup by default for performance reasons"）。
也就是说**所有用 `isaaclab.app.AppLauncher` 启动的脚本天然不受这个问题
影响**——只有像最初的裸 `SimulationApp` smoke test 那种脚本才需要手动加
这个 flag。M4 之后的所有脚本统一走 `AppLauncher`。

## 4. `./isaaclab.sh --install` 的两个坑

### 4.1 torch+cu128 下载极慢

`pip` 从 `download.pytorch.org/whl/cu128` 下载 torch 2.7.0+cu128
（~3.5-4GB）速度只有 9-103KB/s，按此速度要 15-30+ 小时。**用户自行配置好
镜像源后重新执行 `./isaaclab.sh --install` 解决**（不是脚本本身的问题，
是网络/镜像环境问题，无需修改脚本）。

### 4.2 `--install -v` 报错 `isaaclab_rl[-v]` `InvalidRequirement`

```
./isaaclab.sh --install -v
...
InvalidRequirement: Expected end or semicolon (after name and no valid version specifier)
    isaaclab_rl[-v]
```

**根因**：`isaaclab.sh` 的 `--install [rl_framework_name]` 把第二个参数
当作 RL 框架名（用于 `isaaclab_rl[<name>]` extra），脚本本身**不支持**
`-v`/`--verbose` 这类透传给 pip 的参数。

**修复**：去掉 `-v`，直接 `./isaaclab.sh --install`（如需看 pip 详细输出，
没有官方透传选项）。

## 5. URDF 导入：两个版本兼容坑

M4 冒烟测试（导入 `tarantula_core.urdf`，按 docs/01 §7 设置关节驱动
stiffness/damping）暴露了两个 `isaaclab.sim.converters.UrdfConverter`
相关的版本兼容问题：

### 5.1 `ModuleNotFoundError: No module named 'isaacsim.asset'`

`UrdfConverter.__init__` 内部 `from isaacsim.asset.importer.urdf._urdf
import acquire_urdf_interface`，但 `isaacsim.exp.base`（IsaacLab
`AppLauncher` 默认体验）不会自动启用 URDF importer 扩展。

**修复**：在 `import UrdfConverter` **之前**手动启用扩展：

```python
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")
```

### 5.2 `AttributeError: 'ImportConfig' object has no attribute 'set_merge_fixed_ignore_inertia'`

**根因**：IsaacLab 2.3.2 的 `UrdfConverter._get_urdf_import_config()`
假定 URDF importer ≥2.4（有 `set_merge_fixed_ignore_inertia` 方法），但
`isaacsim[extscache]==4.5.0.0` 自带的 URDF importer 是 **2.3.10**（用
`strings <so文件> | grep -i merge` 确认没有该符号）。

**修复**：在 `~/IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`
第 142-143 行附近加 `hasattr` 守卫：

```python
import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)
if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
    # 2.3.10（isaacsim==4.5.0.0 extscache 自带版本）没有这个方法，2.4+ 才有
    import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)
```

**⚠️ 这个 patch 在 tarantula repo 之外**（`~/IsaacLab` 是独立的 editable
install clone，不在本仓库 git 历史里）。如果以后重新 `./isaaclab.sh
--install`、重新 clone `~/IsaacLab`，或升级 isaacsim/isaaclab 版本，**这个
patch 会丢失，需要重新打**（或者升级 `isaacsim[extscache]` 到 ≥2.4 后就不
再需要）。

## 6. 内存监控 wrapper（重 GPU 实验的标准做法）

鉴于本机历史上多次因 GPU 相关问题自动关机，所有新的 Isaac Sim/Lab 实验都
用一个后台脚本包一层，每秒 poll RSS，超过 cap 就 `kill -9`：

```bash
#!/bin/bash
set -uo pipefail
source ~/isaac_venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
python3 -u /tmp/your_script.py > /tmp/your.log 2>&1 &
PID=$!
LIMIT_KB=$((12*1024*1024))  # 12GB RSS 安全上限（15GB 内存 + 15GB swap）
for i in $(seq 1 600); do
  kill -0 $PID 2>/dev/null || break
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ] && [ "$RSS" -gt "$LIMIT_KB" ]; then
    kill -9 $PID; break
  fi
  sleep 1
done
wait $PID 2>/dev/null
```

另外：`python3 -u`（无缓冲）必不可少，否则脚本被 kill 或正常退出时
stdout 缓冲区里的内容（包括最终的 `OK` 标记）可能看不到。

## 7. M4 验收结果（已通过）

用上述全部修复后，`/tmp/m4_smoke_test.py`（基于 `AppLauncher`）跑通：

1. `enable_extension("isaacsim.asset.importer.urdf")`
2. `UrdfConverter` 导入 `tarantula_core.urdf`（裸底盘，`lidar:=false`），
   按 docs/01 §7 映射设置关节驱动：
   `susp_*_joint` → `target_type=position, stiffness=120, damping=8`；
   `wheel_*_joint` → `target_type=none`（自由轮）。
3. `World` + `add_default_ground_plane()` + `Articulation` 加载、`reset()`。
4. 平地站立 8s（PhysX），`susp_*_joint` 在 t=1s 内收敛到：
   `{fl: -0.0529, fr: -0.0529, ml: 0.0511, mr: 0.0511, rl: 0.0493, rr: 0.0493}` rad。
5. 与 Gazebo dartsim 基线 `susp_fl_joint ≈ -0.057 rad` 对比，约 7% 偏差，
   符号/量级一致（归因于 PhysX vs dartsim 物理引擎差异）。

`/tmp` 下的产物（`tarantula_core.urdf`、`tarantula_usd/`、各 smoke test
脚本）会在重启后清空，需要时按上面 §5/§7 重新生成（生成命令见
docs/01 §7："Isaac 导入入口"）。
