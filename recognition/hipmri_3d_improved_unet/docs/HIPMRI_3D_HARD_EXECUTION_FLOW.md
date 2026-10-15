# HipMRI 3D Improved U-Net: 后续执行流程

## 项目边界

本目录是 COMP3710 Hard Difficulty 的正式项目：下采样 HipMRI 3D NIfTI volume 上的分割任务。正式比较对象为 plain 3D U-Net 与包含 volumetric residual units、two deep-supervision heads 的 Improved 3D U-Net。

OASIS 2D Easy 项目已归档于：

`D:\OneDrive - The University of Queensland\Desktop\Study\COMP3710_PatternRecognition\Ass1\OASIS_2D_Easy_Archived_2026-10-15\oasis_brain_analysis`

OASIS 的 Dice、训练曲线、checkpoint、PNG 审计、feasibility 文档及任何 2D 结果均不得写入本项目的 HipMRI 结果表、figures、feasibility evidence 或最终结论。

## 完成定义

本项目在满足以下三项前不得称为完成：

1. 正确的 HipMRI Hard 代码和证据位于向课程上游 `topic-recognition` 分支提交的开放 PR；
2. 同一版本的 README PDF 已在 Turnitin 获得提交确认；
3. 教学团队的后续 PR review 已通过带验证的 commits 和逐条 GitHub 回复处理。

本地代码、合成数据测试、计划文档或 smoke test 都不构成提交确认。

## 阶段 0：确认课程与资源事实

- [ ] 在 Blackboard/2026 course profile 确认截止时间、中期 practical consultation、PR 目标仓库和分支、Turnitin 入口、review 回复期限。
- [ ] 在 Rangpur 检查 `/home/groups/comp3710/HipMRI_Study_open` 是否可读；记录访问日期和目录结构，不复制原始数据到 Git。
- [ ] 向 tutor 确认 label semantics、prostate Dice `>= 0.70` 的计算定义、background 是否单列、允许的 downsampling、GPU 预算和 midpoint 证据要求。
- [ ] 将获得的明确课程信息补入 README 的 project charter；不从旧年份 course profile 推断日期。

**退出条件：** 数据路径、模块难度、目标 metric、提交路径和中期安排均有当前课程来源。

## 阶段 1：HipMRI 3D 数据审计与固定 split

- [ ] 使用 NiBabel 递归匹配 `semantic_MRs` 与 `semantic_labels_only` 下的 NIfTI image/mask pairs。
- [ ] 对每个 pair 验证：patient ID、image-mask shape、affine、orientation、finite intensity、integer labels、空/异常 mask 和重复 relative path。
- [ ] 生成只读 `dataset_manifest.json`，记录 source-relative path、patient ID、shape、spacing、orientation、labels 和 split。
- [ ] 以 patient 为单位生成 seed-42 train/validation/test manifest；自动断言三组 patient IDs 没有重叠。
- [ ] 输出至少三位真实患者的 axial、coronal、sagittal 图像与标签叠图；mask resize 必须使用 nearest-neighbour interpolation。

**退出条件：** 真实 audit report、sample panels 和可重复 patient-disjoint manifest 已保存；没有未知标签、坏配对或跨 split patient。

## 阶段 2：确定 3D 预处理与计算预算

- [ ] 仅从 train patients 推导 intensity normalisation 参数，并记录实现和统计值。
- [ ] 明确并固定 downsampled volume geometry、voxel spacing、orientation policy、label interpolation policy、patch size、stride、overlap、foreground sampling rule 和 batch size。
- [ ] 在 Rangpur GPU 上执行一个小型 3D baseline smoke run，保存命令、GPU 型号、CUDA/PyTorch 版本、VRAM、stdout、stderr、job ID 和 exit state。
- [ ] 若完整 volume 超出显存，使用 overlap patch inference，并验证 logit accumulation 后每一个 voxel 的 prediction count 大于零。

**退出条件：** 选定的 geometry 和 batch size 在真实 HipMRI volume 上完成 forward、backward、optimizer step、checkpoint save/load 与小型 inference。

## 阶段 3：建立公平的 3D baseline

- [ ] Baseline 使用 plain 3D U-Net；输入为 `(N, C, D, H, W)`，输出为 `(N, classes, D, H, W)`。
- [ ] 测试 forward shape、finite loss、nonzero gradients、optimizer step、checkpoint reload 和 full-volume reconstruction。
- [ ] 训练只使用 train patients；每个 epoch 在 validation patients 计算主 selection metric。
- [ ] 只按 validation metric 选择 `best.pt`；禁止用 test results 选 epoch、loss、patch size 或模型配置。
- [ ] 保存 baseline 的 config、split manifest、history、best checkpoint、test metrics、resource record 和 predictions 到 Git 忽略的实验目录。

**退出条件：** baseline 在固定 manifest 上完成完整训练，并能从 `best.pt` 重建最终 held-out test 结果。

## 阶段 4：完成 Improved 3D U-Net

- [ ] 以 `Conv3d`、3D normalisation、3D residual shortcut 实现 volumetric residual units；不可复用 2D layers 代替。
- [ ] 在 decoder 设置两个 deep-supervision heads，并将其 logits 对齐至对应 target geometry。
- [ ] 记录总损失：`main_loss + lambda_1 * aux_loss_1 + lambda_2 * aux_loss_2`；将 `lambda_1` 和 `lambda_2` 固定在 config。
- [ ] 测试 auxiliary output shapes、weighted loss、backpropagation、inference 仅返回 final-resolution logits 和 checkpoint reload。
- [ ] baseline 与 improved 使用同一 manifest、相同预处理、同一 metric 实现、同一 validation selection rule 和同一 test protocol。

**退出条件：** Improved 3D U-Net 的每一项 Hard 要求由源代码、测试和真实 GPU smoke evidence 共同支持。

## 阶段 5：正式实验与结果保存

- [ ] 对 baseline 和 improved 分别运行完整训练；每个实验保存不可变 config、seed、hardware、job ID、stdout、stderr、history、best checkpoint 和 test metrics。
- [ ] 在最终 `best.pt` 上执行一次 held-out test；不要在读取 test metric 后继续调参并重复报告最佳 test 值。
- [ ] 对每一类别及 case-level distribution 报告 Dice 和 IoU；同时报告 macro/foreground metric，并将 background 单独列示。
- [ ] 报告 parameter count、peak VRAM、total training time、per-patient inference time、patch/full-volume inference setting。
- [ ] 判断 `>= 0.70` 时注明计算定义、患者数量、类别和是否排除 background。

**退出条件：** 两模型的数值来自可定位的 config、manifest、checkpoint、log 和机器可读 metric 文件。

## 阶段 6：失败案例与工程建议

- [ ] 按预先声明的 case-level error statistic 对 held-out patients 排序，选择 3–5 个真实 failure cases；不只展示最好样本。
- [ ] 每个案例输出 axial、coronal、sagittal MRI、ground truth、baseline、improved、false-positive/false-negative overlay 和 per-class metrics。
- [ ] 对每例说明失败部位、错误类型、improved 相比 baseline 的变化和人工复核是否可接受。
- [ ] 结论仅可在证据允许的范围内选择：推荐 improved 作为人工审阅草图；推荐 baseline；或列出明确的后续改进目标。不得主张 autonomous clinical deployment。

**退出条件：** README 中的图、数字、病例解释都可追溯到 held-out HipMRI evidence。

## 阶段 7：中期与提交

- [ ] 基于真实 HipMRI audit 和 smoke evidence 重写一页 feasibility review；不得使用 OASIS 数值或图。
- [ ] 在 scheduled practical consultation 出示该 review，并记录 tutor 已确认的日期、反馈和 scope decision。
- [ ] 从干净 shell 执行 README 写明的 install、test、train 与 prediction 命令；检查没有数据、checkpoint、credentials 或大临时文件进入 Git。
- [ ] 导出最终 README PDF，逐页检查 figure legibility、table overflow、heading order、references 和 AI usage disclosure。
- [ ] 创建从个人 fork `topic-recognition` 到课程上游 `topic-recognition` 的 PR，保存 URL 和状态记录。
- [ ] 提交匹配 README 的 PDF 至 Turnitin，保存 receipt、submission ID 和 timestamp。

## 阶段 8：教学团队代码审查

- [ ] 阅读每条 PR review comment；先复现问题或确认文档请求的范围。
- [ ] 对每个 coherent fix 创建聚焦 commit，并运行相关测试。
- [ ] 逐条回复 GitHub review thread，引用 commit SHA，说明修复和验证方式。
- [ ] 在课程规定窗口结束前保存 review discussion 的最终状态。

## 不可突破的证据规则

- OASIS 2D 数据、结果和图不能证明 HipMRI 3D Hard 项目。
- 合成 NIfTI 测试证明代码路径，不证明真实 HipMRI performance。
- GPU smoke run 证明训练链路，不证明 Dice 达标。
- validation 用于 checkpoint selection；held-out test 只用于最终一次正式评估。
- 本地文件、push 或草拟 PR 不等于 Turnitin 提交确认。
