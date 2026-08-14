# Jetson -> catas Codex 接力

接力机器：`catas@192.168.201.133`  
接力会话：`019fb082-00d4-7c50-99a6-0cf7d5287475`

## 用户最终授权

用户要求把原 goal 的全部剩余工作交给新电脑上已经可用的 Codex 执行。
Jetson 随后只做敏感信息清理，不再承担后续迁移判断或构建工作。

## 已完成并有现场证据

- 新机 Codex 0.146.0 x86_64，`codex login status` 为 ChatGPT 官方登录。
- `codex-chatgpt-tunnel.service` 已启用且 active。
- dstopology/code4ai 两套 provider 在新机通过隧道得到 HTTP 200。
- 62/62 Codex session、66 个 memory 文件、SQLite 状态与 `/resume` 已验证。
- 当前 goal 会话已在 2026-07-30 12:18 再次增量同步、适配为
  `/home/catas` + `openai`，`CODEX_STATE_VERIFY_OK`。
- Elfin/视觉源码（含未提交和未跟踪文件）、RealSense ROS 源码、
  `elfin_citrus_data`、关键 `.ros` 标定/载荷/自由拖动/采摘数据、
  完整 `.ros` 历史归档、远端推理 token、依赖/Git/SHA 清单均已复制。
- 12 个原名启动入口位于 `/home/catas`；33 个远端副本文件的旧主目录
  已原子改为 `/home/catas`，备份在
  `/home/catas/migration_backups/path-adaptation-20260730-035111`。
- 桌面已有：
  - `Codex登录切换与应急恢复.txt`
  - `Elfin启动命令.txt`
  - `迁移状态与首次连接检查.txt`
- 静态验证已通过文件、Shell、Python 和路径检查；唯一当时门槛是宿主
  Ubuntu 18.04 / ROS Melodic 没有 Noetic。

## 正在新机独立运行

`/home/catas/migration_from_jetson/noetic/install_noetic_proot.sh`

它在 `/var/tmp/catas-robotics/noetic` 安装隔离 Ubuntu 20.04 / ROS Noetic，
不修改宿主 Melodic。截至交接时，基础 Focal/ROS 仓库已完成，正在安装
Desktop Full、MoveIt、Gazebo 11、SOEM、RealSense、OpenCV/PCL 和 Python 3。

先检查：

```bash
pgrep -af 'install_noetic_proot|apt-get install|dpkg'
find /var/tmp/catas-robotics/noetic/markers -maxdepth 1 -type f -printf '%f\n' | sort
```

不要并行启动第二个 apt。原进程结束后若没有
`noetic-packages-installed`，读取安装输出/apt 状态并原地续跑同一脚本。

## Jetson 最后一批大目录传输

Jetson 正在把完整 9.5 GB `ros2_ws`（包括 ORB-SLAM2、YOLO_ORB_SLAM3、
x86_64 CUDA 11.8 libtorch 压缩包与历史 build/devel）送往：

`/var/tmp/catas-robotics/migrated/ros2_ws-full-jetson`

完成标记：

`/var/tmp/catas-robotics/migrated/ROS2_FULL_TRANSFER_OK.txt`

活动 RealSense 工作区已经在 `/home/catas/ros2_ws`，不要用历史 ARM64
build/devel 覆盖它。完整镜像只作保存和择取源码。

## 必须继续完成的原 goal

1. 等待/修复 Noetic 安装，验证：
   `catas-noetic bash -lc 'source /opt/ros/noetic/setup.bash; rosversion -d'`。
2. 在隔离 Noetic 中运行：
   `catas-noetic /home/catas/migration_from_jetson/noetic/build_robotics_noetic.sh`。
3. 修复真实构建暴露的依赖/源码问题，运行项目单元测试和
   `catas-noetic /home/catas/TEST_ELFIN_OFFLINE.sh`；禁止启动 EtherCAT。
4. PRoot 只用于构建/仿真。为真实 E05 准备宿主 Ubuntu 20.04 或具备
   raw network、设备、RT/cgroup 权限的 Noetic 容器方案。任何 sudo
   安装先生成并审计脚本，让用户在新机本地执行；不做 broad upgrade，
   不卸载 Melodic。
5. 用户挂载 4.6 TB Ling 盘后，把 `/var/tmp` 完整镜像、迁移归档和大型
   模型转存到外盘工作归档；外盘未挂载前不格式化、不删除源副本。
6. 修复新机 NVIDIA 驱动（当前 PCI 可见但 `nvidia-smi` 失败）后，在
   x86_64/CUDA 11.8 上重建本地 TensorRT/Conda 环境；不可复用 Jetson
   ARM64 engine/env。
7. 验证硬件、Panel、视觉三个原命令的环境与参数解析；真实硬件验证严格按
   静态配置 -> 构建 -> 仿真 -> 传感器 -> 低速无负载 -> 集成顺序。
   没有用户当次明确授权、实体急停、清场和守闸，绝不发运动命令。
8. 完成低优先级附属软件/udev/远程桌面兼容检查并更新桌面状态文档。
9. 做逐项完成审计；只有原 goal 每一项有当前证据时才标记 complete。

## 接力 Codex 行为

- 开始先调用 `get_goal`；若本会话没有活动 goal，再按原 objective
  调用 `create_goal`。
- 保持中文、保姆级说明和不超过约 60 秒的进度更新。
- 不要为了省事把“可构建”冒充“可连接真实机械臂”。
- 不要尝试回头恢复 Jetson 上随后被用户要求删除的 sessions/memory/密钥。
