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
| label mapping | `0=Background, 1=Body, 2=Bones, 3=Bladder, 4=Rectum, 5=Prostate` | `evidence/SOURCES.md`、CSIRO v2 DOI `10.25919/45T8-P065` 的 `Labels:` block | 任意模型性能或临床可用性 |
| 现有代码 | 数据审计、模型、训练 CLI、validation checkpoint、checkpoint restore、test-only sliding-window inference、JSON metrics、ranking 与多平面 failure panels | Python 源码、测试 | 真实 GPU、正式 checkpoint 或 Dice 已达标 |
| 本地测试 | 2026-10-21：`python -m pytest -q` 为 `26 passed` | 本地输出 | 真实 GPU、checkpoint 或 Dice 已达标 |

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
- failure cases 来自 held-out HipMRI 低表现样本。比较工具按共同 FP/FN 错误分数排序，并在 GT 前列腺范围内取误差最大的切片；不得把未经方向与解剖验证的数组索引称为 prostate base。

### 2.3 已冻结的 label mapping 与仍然适用的停止条件

`evidence/sources/csiro-brief-data-description.txt` 是 CSIRO collection v2（DOI `10.25919/45T8-P065`）的本地冻结 supporting file。其 `Labels:` block 明确规定：`0=Background, 1=Body, 2=Bones, 3=Bladder, 4=Rectum, 5=Prostate`。文件 SHA-256 为 `DC3A99204684CED859872D79A75707AF3C0AD31D00F5F3213063CE5E30727EE6`；正式 metadata 使用 `prostate_label=5` 与 `evidence/SOURCES.md` 所列的完整来源文字。

不得以编号、频率、mask 体积或目测替换该来源。checkpoint 缺少 label 或非空 source 时，headline prostate Dice 评估必须拒绝执行；有该 mapping 也不等于已经得到任何真实 Dice。

## 3. 修正后的执行阶段

### A. 冻结已完成的数据事实

- [x] 递归发现 image/mask NIfTI pairs。
- [x] 验证 shape、affine、orientation、finite image values 与 integer masks。
- [x] 生成 manifest，固定 patient-disjoint seed-42 split。
- [x] 核验 211 cases / 38 patients，患者分组 27/6/5。
- [ ] 保存至少三位真实患者的 axial/coronal/sagittal 图与标签叠图。`render_case_views()` 可生成面板，但当前没有真实数据产物。
- [x] 记录 CSIRO collection v2 的权威 label-to-structure mapping；`5=Prostate`。

### B. 训练程序：已具备本地可执行链路与真实 A100 smoke 证据，尚缺正式双模型实验

`train.py` 已有 CLI、manifest-driven DataLoader、Adam、epoch loop、validation-only checkpoint selection、`experiment_config.json`、`history.json`、`best.pt` 与 `training_summary.json`，并拒绝覆盖非空 artifact directory。checkpoint 内保存模型配置、manifest SHA-256 与 prostate label/source；`load_checkpoint_model()` 可按其声明恢复 baseline 或 improved architecture。

- 训练只使用 train patients；validation 选择 best checkpoint；test 不参与 epoch、patch size、loss 或配置选择。
- 正式 Slurm 运行仍须补充 Git SHA、Python/PyTorch/CUDA、GPU、stdout/stderr、job ID、exit state 与峰值显存；这些环境记录尚未由当前训练 CLI 自动写入。
- 当前实现是每-volume z-score 和固定 `target_shape`；“仅从 train patients 拟合 normalisation 参数”尚未实现，不得写成现状。
- Rangpur jobs `680366` 与 `680381` 已覆盖 forward、backward、optimizer step、checkpoint save/load 与 held-out inference；两者均为一 epoch smoke，不能作为正式性能结论。

### C. 预测与患者级指标：已有真实 smoke 产物，尚缺新版正式资源证据

`predict.py` 已从 validation-selected `best.pt` 恢复声明的 architecture，验证 manifest SHA-256，且只读取 manifest 的 test patients。它对每个 volume 做 overlap-averaged sliding-window logits 重建，按患者计算 prostate Dice 的算术均值，写出 `test_metrics.json`、非背景 label Dice/IoU、failure ranking 和每名 test patient 的代表性多平面 panel。合成 NIfTI 端到端测试证明该命令不会访问训练/验证病例。

正式评估应从 validation-selected `best.pt` 开始，对每位 held-out patient 的完整 volume 重建，然后输出：

1. 每位患者的已确认 prostate label Dice；
2. `mean_patient_prostate_dice` 与 test patient count；
3. 每个非 background label 的 Dice/IoU；
4. 参数量、peak VRAM、总训练时间与 per-patient inference time；新版 CLI 已写入这些字段，下一次正式 run 才会产生真实新版 artifact；
5. 按预先声明的患者级 prostate error 排序的 3–5 个真实 held-out failure panels。

必须新增测试，证明患者 Dice 算术均值不能被 pooled-voxel Dice 替代；baseline 与 improved 使用同一实现。

### D. 公平比较与 GPU 证据

- 两模型共享 manifest、preprocessing、target geometry、patch/stride/overlap、metrics、validation-only selection rule 与 test protocol。
- Rangpur 或经 staff 批准的替代环境保存 GPU model、CUDA/PyTorch versions、command、job/session ID、exit state、stdout、stderr 和 peak VRAM。
- 先完成 baseline GPU smoke，再运行完整 baseline/improved。

### E. 当前 preflight 与抽样证据边界

- `683036` 的 `PD (Priority)` 仅表示调度等待。不得重复提交同类 preflight，也不得用本地缩小 geometry 的训练结果替代其 `128³`、A100、200 optimizer updates 证据。
- 作业运行或结束后，读取 `artifacts/rangpur_preflight_683036/git_sha.txt`。其 SHA 不等于当前预期提交时，产物只可用于排错，不能用于决定正式配置。
- `configs/hipmri_experiment_candidates.json` 是唯一共享超参数来源：`preflight-a` 固定 CE/Dice 权重 `4.0/4.0`、前景采样概率 `0.50`；`formal-b` 固定 `8.0/8.0` 与 `0.75`。两者共享 `128³` target、`96³` train patch 与 seed 42。模型名称不属于共享候选配置：preflight 明确使用 Baseline，正式脚本以其单独记录的 `HIPMRI_MODEL` 指定 Baseline 或 Improved。
- 本地 `B006_Week0` patch audit 已在 `128³ → 96³`、每个概率 200 次抽样下完成。`p=0.50` 与 `p=0.75` 都有 100% patch 含 prostate，均值均为 2552 个 prostate voxels；因此该病例不能证明提高采样概率会增加正例暴露。
- `preflight_summary.json` 现受端到端契约测试约束，包含 patient/case、manifest SHA、标签来源、几何、权重、采样概率、固定 optimizer updates、评估间隔、TP/FP/FN 与 Dice。默认诊断为 200 updates、每 20 updates 评估一次；正式 Baseline、Improved、held-out 指标和同病例 panels 仍未完成。
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

1. Virtual GPU Lab 是否可用于正式完整训练及需要保存的证据；
2. Phase 1 GitHub PR 与 Turnitin README PDF 的确切日期和时间。

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
