# catas 真实运行环境准备（不等于 EtherCAT 已就绪）

这组文件把“可以在 PRoot 编译”与“可以连接真实 Elfin E05”明确分开。
所有默认动作都是只读审计或安装模拟；安装脚本只有显式 `--apply` 才会修改
系统，而且没有 `apt upgrade`、没有卸载 Melodic、没有自动重启。

## 2026-07-30 当前硬门槛

- 当前启动的是 Ubuntu 18.04 / kernel 4.15，不是原生 Noetic/Focal。
- NVIDIA PCI 设备 `10de:28e0` 可见，但没有 NVIDIA 内核模块，
  `nvidia-smi` 失败；本机已有 CUDA 11.8 工具链和 x86_64 TensorRT
  8.5/8.6 文件，它们仍需正确驱动后才能构建 engine。
- 唯一物理 Ethernet `enp8s0` 正承载 `192.168.201.133/23` 管理网络，
  不能把它当 EtherCAT 口。真实 E05 需要一块物理上可辨识的专用网卡。
- 当前内核 `CONFIG_RT_GROUP_SCHED` 未启用，现有硬件启动脚本所需的
  `cpu.rt_runtime_us` 不存在。不能绕过；应在最终系统上重新审计实时内核、
  `chrt`、cgroup 和 `cyclictest`。
- Ling 外盘是 UUID `D09A-E492` 的 exFAT，但宿主缺少 exFAT helper，
  正常挂载报 `unknown filesystem type 'exfat'`。
- 当前 Docker 官方安装文档和 NVIDIA Container Toolkit 当前测试矩阵都
  不再覆盖 Ubuntu 18.04。因此不能把在 Bionic 上临时拼出的容器称为真实
  生产就绪方案。

## 推荐顺序

1. 先在当前机器运行只读审计：

   ```bash
   /home/catas/migration_from_jetson/real_runtime/audit_host_no_motion.sh
   ```

2. 等隔离 Noetic 安装和构建完全结束后，在本机终端先模拟再安装 exFAT
   支持；这只增加两个精确包：

   ```bash
   /home/catas/migration_from_jetson/real_runtime/install_exfat_support_bionic.sh --simulate
   sudo /home/catas/migration_from_jetson/real_runtime/install_exfat_support_bionic.sh --apply
   udisksctl mount -b /dev/sda1
   ```

3. 挂载后先计划、再复制归档。脚本用 tar 保存 ext4 的链接/权限语义，
   校验 SHA-256，且没有任何删除命令：

   ```bash
   /home/catas/migration_from_jetson/archive/archive_to_ling.sh --plan
   /home/catas/migration_from_jetson/archive/archive_to_ling.sh --apply
   ```

4. NVIDIA 驱动同样先模拟；安装后由用户选择合适时机重启，脚本不重启：

   ```bash
   /home/catas/migration_from_jetson/real_runtime/install_nvidia_driver_530_bionic.sh --simulate
   sudo /home/catas/migration_from_jetson/real_runtime/install_nvidia_driver_530_bionic.sh --apply
   ```

5. 真实 E05 主路径是单独的原生 Ubuntu 20.04 安装（单独 SSD/已明确规划
   的分区），保留当前 18.04/Melodic，不在本文件里做分区或 bootloader
   操作。启动那个系统后才允许审阅并运行：

   ```bash
   /home/catas/migration_from_jetson/real_runtime/prepare_native_focal_noetic.sh --audit
   sudo /home/catas/migration_from_jetson/real_runtime/prepare_native_focal_noetic.sh --apply
   ```

## 真实硬件仍需逐项通过

1. 原生 Noetic 构建、全部单元测试和离线 launch/URDF 检查。
2. 专用 EtherCAT 网卡的物理标签、无 IP/无默认路由，以及准确的 4 从站
   身份。此时才可由现场人员授权只读 `slaveinfo`。
3. 最终内核的实时配置、RR 10 权限和至少一次独立 `cyclictest` 记录。
4. 实体急停/电闸、空载、清场和专人守闸。
5. 先 Servo Off 读取状态，再仿真/传感器，之后才是用户当次明确授权的
   低速无负载单步；没有该次授权绝不发运动命令。

PRoot 的构建结果和这些准备脚本都不证明 EtherCAT、I/O、Servo On 或运动
已经可用。
