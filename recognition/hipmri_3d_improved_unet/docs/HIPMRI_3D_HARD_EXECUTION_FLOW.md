# HipMRI 3D Improved U-Net：执行总表、课程说明与证据边界

> **适用范围：** COMP3710 Assignment 1，`recognition/hipmri_3d_improved_unet/`。本文件合并项目执行路线与 Ed Discussion 中对本项目有效的补充信息。
>
> **优先级：** 课程 brief、Blackboard、course profile 与教学团队后续公告高于本文件；已保存的 Ed 回复高于旧计划；代码、测试和 manifest 仅证明本地已核验状态。
>
> **边界：** 本文件不等于 tutor check-off、Rangpur job、PR、Turnitin 或正式实验结果。OASIS 2D 数据、曲线、checkpoint、图和 Dice 不得写入 HipMRI 3D Hard 的结果、feasibility evidence 或结论。

## 1. 当前经过核验的事实

| 项目 | 当前状态 | 依据 | 不可由此推出 |
|---|---|---|---|
| 项目范围 | HipMRI 3D Improved U-Net，Hard difficulty；比较 plain 3D U-Net 与 residual + deep-supervision Improved U-Net | `README.md`、`modules.py`、课程报告 | 正式比较已完成 |
| 数据审计 | 211 volumes、38 patients、labels `0–5`，路径为相对路径 | `evidence/dataset_manifest.json` | labels 的医学语义 |
| 患者隔离 | seed 42；train/validation/test 为 27/6/5 patients、146/46/19 volumes | manifest、`dataset.py`、`audit.py` | 模型没有其他设计问题 |
| 现有代码 | 数据审计、模型、损失、logit overlap averaging、多平面绘图函数 | Python 源码 | 已有可运行训练或预测程序 |
| 本地测试 | 2026-10-21：`python -m pytest -q` 为 `14 passed` | 本地输出 | 真实 GPU、checkpoint 或 Dice 已达标 |

## 2. 已确认的课程与 staff 约束

### 2.1 课程报告

- 只完成一个选定项目；当前正式项目是 `recognition/hipmri_3d_improved_unet/`。
- 3D Improved U-Net 是 Hard difficulty，需 volumetric residual units 与 deep supervision；课程表给出 test-set Dice `>= 0.70` 目标。
- 必须提供一页 feasibility review 并完成 practical consultation check-off。check-off 是通过 assessment 的硬性条件，README 必须保留该 review。
- 必须提供 `modules.py`、`dataset.py`、`train.py`、`predict.py`、测试、README、渐进 commit history、PR 与 Phase 2 review 回复。
- 不提交数据或模型文件；Turnitin 提交 README 的 PDF 导出；README 必须包含 **Artificial Intelligence Usage Disclosure**。

### 2.2 Zhao Wang 的针对性回复（2026-10-21）

- plain 3D U-Net 可作为 baseline。
- 211 volumes 来自 38 patients；两个模型均按 **patient** 而非 volume、slice 或 patch 切分。
- headline 是 held-out test patients 的 prostate Dice 算术平均：每位患者先 full-volume reconstruction，再计算 Dice，最后求患者均值。
- `>= 0.70` 适用于该患者级 prostate Dice mean，不能改为 pooled-voxel Dice、foreground macro Dice 或全类别均值。
- 报告所有非 background labels 的 Dice；background 不报告。
- sliding-window inference 可用，但两个模型共用 split、preprocessing、downsampling 和 evaluation protocol；记录 patch size、overlap 与 logit merging method。
- failure cases 来自 held-out HipMRI 低表现样本，优先 prostate base。

### 2.3 未确认的 label mapping：硬性停止条件

`semantic_labels_only` 的 `0–5` 尚无可定位的 prostate label mapping。不得以编号、频率、mask 体积或目测推断 prostate index。

在 HipMRI reference paper 或权威数据说明给出页码、表、图或等价可定位来源前：

- 不得把任一 label 写成 prostate；
- 不得将任一数值称为 prostate Dice；
- 不得宣称 `>= 0.70`；
- prostate-headline 评估必须拒绝缺失 mapping/source 的配置。

## 3. 修正后的执行阶段

### A. 冻结已完成的数据事实

- [x] 递归发现 image/mask NIfTI pairs。
- [x] 验证 shape、affine、orientation、finite image values 与 integer masks。
- [x] 生成 manifest，固定 patient-disjoint seed-42 split。
- [x] 核验 211 cases / 38 patients，患者分组 27/6/5。
- [ ] 保存至少三位真实患者的 axial/coronal/sagittal 图与标签叠图。`render_case_views()` 可生成面板，但当前没有真实数据产物。
- [ ] 获取并记录 label-to-structure mapping 的权威来源。

### B. 补齐可执行训练程序

`train.py` 当前仅有 manifest 读取、组合损失、批级指标与 validation checkpoint 选择工具，缺少 CLI、DataLoader、optimizer、epoch loop、history、checkpoint、不可变 config 和环境记录。

- 训练只使用 train patients；validation 选择 best checkpoint；test 不参与 epoch、patch size、loss 或配置选择。
- 保存 config、seed、manifest hash、Git SHA、Python/PyTorch/CUDA、GPU、stdout/stderr、history、best checkpoint 与资源记录。
- 当前实现是每-volume z-score 和固定 `target_shape`；“仅从 train patients 拟合 normalisation 参数”尚未实现，不得写成现状。
- 真实 GPU smoke 需覆盖 forward、backward、optimizer step、checkpoint save/load 与小型 inference。

### C. 补齐完整 volume 推理与患者级指标

`predict.py` 已实现 overlap-averaged logits：累加每个 patch 的 logits，再以 `prediction_count` 确认每个 voxel 被覆盖。尚缺 checkpoint loading、test orchestration、JSON metrics、case ranking 和真实 failure panels。

正式评估应从 validation-selected `best.pt` 开始，对每位 held-out patient 的完整 volume 重建，然后输出：

1. 每位患者的已确认 prostate label Dice；
2. `mean_patient_prostate_dice` 与 test patient count；
3. 每个非 background label 的 Dice/IoU；
4. 参数量、peak VRAM、总训练时间与 per-patient inference time；
5. 按预先声明的患者级 prostate error 排序的 3–5 个真实 held-out failure panels。

必须新增测试，证明患者 Dice 算术均值不能被 pooled-voxel Dice 替代；baseline 与 improved 使用同一实现。

### D. 公平比较与 GPU 证据

- 两模型共享 manifest、preprocessing、target geometry、patch/stride/overlap、metrics、validation-only selection rule 与 test protocol。
- Rangpur 或经 staff 批准的替代环境保存 GPU model、CUDA/PyTorch versions、command、job/session ID、exit state、stdout、stderr 和 peak VRAM。
- 先完成 baseline GPU smoke，再运行完整 baseline/improved。
- 实验产物放 Git 忽略目录；不提交 data、weights、checkpoints 或大临时文件。

### E. Feasibility、提交与 code review

- feasibility review 使用 HipMRI audit、患者隔离、模型设计、实际 smoke evidence、风险、预算与 fallback；不能使用 OASIS 结果。
- tutor 在 scheduled practical consultation 记录 check-off；本地文件不等于 check-off。
- 从干净 shell 运行 README 的 install、test、train、prediction；确认 data、credentials、weights 与产物不进 Git。
- 导出 README PDF 并检查图、表、引用、标题和 AI disclosure。
- Phase 1：从个人 fork 的 `topic-recognition` 向课程上游同名分支发 PR；保存 URL 与状态。
- Phase 2：复核每条 review comment，做聚焦 commit、运行验证，并在 review thread 引用 commit SHA 回复。

## 4. Ed Discussion：保留的来源与待问问题

| 主题 | 已知要求 | 项目处理 |
|---|---|---|
| 仅完成一个项目 | 只做一个选定模型 | HipMRI 3D Hard 是唯一正式项目 |
| failure cases | 为 dataset 低表现样本，解释小目标、边界、anatomy、质量或标签问题 | held-out test patients，优先 prostate base |
| check-off | 可参加其他 practical，建议及早完成 | 带一页 review 与真实 audit/smoke evidence |
| Virtual GPU Lab | 有 quota，空闲 session 会结束 | 需 staff 确认后才作为 Rangpur 替代，并备份证据 |

仍须向教学团队确认：

1. `0–5` label-to-structure mapping 的可引用位置；
2. Virtual GPU Lab 是否可用于正式完整训练及需要保存的证据；
3. Phase 1 GitHub PR 与 Turnitin README PDF 的确切日期和时间。

**标签映射问题草稿：**

```text
Hi teaching team,

I am implementing Project 2.2 at Hard difficulty: a downsampled HipMRI 3D Improved U-Net with volumetric residual units and deep supervision, compared against a plain 3D U-Net using a patient-disjoint split.

Could you please point me to the exact page, table, figure, or authoritative dataset documentation that maps the course-provided HipMRI semantic labels `0–5` to anatomical structures, specifically the prostate label? I will record that source and mapping in the immutable experiment configuration and report.

Thank you.
```

在现有 due-date 帖子下追问 GitHub PR 与 Turnitin README PDF 的确切 Phase 1 date/time；不从旧年份 course profile 推断。

## 5. 不可突破的证据规则

- 合成 NIfTI 测试、计划文档和本地 smoke 不证明 HipMRI 3D 最终性能。
- validation 用于 checkpoint selection；held-out test 只做最终正式评估，不能在读取后继续调参。
- GPU smoke 证明链路，不证明 Dice 达标或 improved 更好。
- push、草拟 PR 或本地 PDF 不等于上游 PR 与 Turnitin 提交确认。

## 6. 原始材料保留说明

- `COMP3710_Module2_2_3D_Improved_UNet_完整执行指南_中文.md` 保持完整，作为从零到 Phase 2 的详细指南；本总表不替代其操作细节。
- `ED_DISCUSSION_2026-10-18_PROJECT_GUIDANCE.md` 保留原始链接、针对性答复与问题草稿；本总表只汇总其可执行约束。
- `COMP3710_Report_2026_draft_v2.3.pdf` 的文件名为 v2.3，但 PDF 首页内部标题写为 Version 2.2。记录该资料不一致，并以后续课程公告或 Blackboard 为准。
