# CHIP → OLI HU_D04 夹爪版

本次迁移增加独立 OLI 训练和 MuJoCo 部署入口，继续保留 G1 CHIP。

## 范围

- 使用 `robot_assets/oli/` 的 HU_D04 等效串联模型，夹爪关节固定。
- 31 个身体关节（含头部），`nq=38, nv=37, nu=31`。
- CHIP 腿/腰命令仍为 15 关节的位置和速度；历史长度为 10。
- OLI actor 输入 1059 维，输出 31 维；G1 仍为 999/29。
- 三个交互点为 `waist_pitch_link`、左右 `wrist_roll_link`。
- PD、动作尺度、armature、默认站姿从同一 `robot.json` 生成。
- 保留地面碰撞，初版关闭自碰撞；不含真实闭链传动与 OLI 真机 SDK。

原始来源、固定提交和许可证见 `robot_assets/oli/PROVENANCE.md`。
G1 的 CSV/NPZ/checkpoint 与 OLI 不兼容；ADAPT 的 OLI checkpoint 也不是 CHIP checkpoint。

## 同步到 Linux

将更新的本仓库放入 `~/workspace/chip/chip-oli`，然后执行：

```bash
cd ~/workspace/chip/chip-oli
bash scripts/apply_overlay.sh ../mjlab ../RoboJuDo
```

覆盖脚本只复制代码和模型资产，不删除 `.venv`、日志、动作和 checkpoint。
源码仍基于 `metadata/upstreams.txt` 指定的提交。使用之前安装好的独立环境。

## 先验证模型、配置和站姿动作

```bash
cd ~/workspace/chip/mjlab
uv run --no-sync list-envs
uv run --no-sync pytest -q tests/test_oli_chip_task.py \
  tests/test_chip_command.py tests/test_chip_impulse_event.py tests/test_tracking_task.py

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m mjlab.scripts.oli_motion \
  --standing-seconds 10 --output ../assets/motions/oli_standing_smoke.npz

CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m mjlab.scripts.check_oli_chip \
  --motion-file "$HOME/workspace/chip/assets/motions/oli_standing_smoke.npz"
```

站姿文件只验证模型、FK、维度、训练和部署链路，不是步行动作，也不保证站立平衡。
转换器拒绝覆盖已有输出文件；再次生成时使用新的文件名。

若旧配置在检查环境时出现 `nconmax overflow (nconmax must be >= 36)`，
重新应用 overlay 后直接重跑 `check_oli_chip`，无需重新生成已有 NPZ。
OLI 的训练和 play 配置将接触容量设为 `nconmax=128`、约束容量设为
`njmax=1024`；继承的 tracking 配置原先仅允许 35 个接触。

## 10 次 PPO 迭代检查

```bash
uv run --no-sync train Mjlab-Tracking-Flat-LimX-OLI-HU-D04-CHIP \
  --gpu-ids "[1]" \
  --env.commands.motion.motion-file "$HOME/workspace/chip/assets/motions/oli_standing_smoke.npz" \
  --env.scene.num-envs 32 \
  --agent.max-iterations 10 --agent.save-interval 10 \
  --agent.logger tensorboard --agent.run-name oli_smoke

find logs/rsl_rl/oli_chip_tracking_attn -name 'model_*.pt' -print
```

## 部署回放

```bash
cd ~/workspace/chip/RoboJuDo
source .venv/bin/activate
# 查看器已安装时运行；后续重新安装子模块后也可以再次运行。
python scripts/patch_chip_viewer.py

# 将 /实际运行目录/ 替换为上一步生成的具体日志目录。
python scripts/run_chip_pipeline.py -c oli_chip \
  --checkpoint-file /实际运行目录/model_9.pt \
  --motion-file "$HOME/workspace/chip/assets/motions/oli_standing_smoke.npz" \
  --start-mode mjlab --headless --steps 100
```

去掉 `--headless --steps 100` 打开查看器。默认动作结束时间从 NPZ 推导。
CLI `--mouse-force-drag` 继续支持鼠标扰动；实测力响应与策略质量需要充分训练后另行验证。

## 真正的 OLI 动作数据

必须先将人体动作重定向到 OLI，不能把 G1 的 29 关节 CSV 补零当成 OLI。
输入 CSV 无表头，38 列：`root XYZ + root quaternion XYZW + 31 joint positions`。
关节顺序严格为 `robot_assets/oli/robot.json` 的 `joints` 顺序。
`--joint-names-file` 接受该格式的元数据 JSON，或一个关节名称列表 JSON。

```bash
cd ~/workspace/chip/mjlab
CUDA_VISIBLE_DEVICES=1 uv run --no-sync python -m mjlab.scripts.oli_motion \
  --input-file /path/to/oli_retargeted.csv \
  --joint-names-file /path/to/csv_joint_order.json \
  --input-fps 30 --output ../assets/motions/oli_walk.npz
```

转换在本地完成，不上传 W&B。NPZ 保存 robot、fps、关节名称、刚体名称及 FK 数据。
训练与部署均拒绝错误机器人、错序、缺失名称及不同控制频率的数据。
接入真实 OLI 动作后再扩大训练规模，评估动作跟踪和外力交互。

## 验证边界

本机 CPU 检查：`python -m pytest checks/test_oli_migration.py -q`。
覆盖模型维度、关节/PD/armature 映射、有限值步进、G1/OLI 训练-部署注意力
数值一致性以及错误动作文件拒绝。Linux 固定版本环境的完整转换、环境步进、
PPO 与 RoboJuDo 回放需执行上述命令；本机检查不代表行走性能或真机验证完成。
