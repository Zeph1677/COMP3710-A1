# Local reference learning record — 25 October 2026

This record inspects local `tmp/s4699027_task_3D_uNet` as a learning reference. It is not a course requirement, independently reproduced benchmark, or source of this project's results. The folder name and source-header author text differ; neither is used to infer identity or ownership. Reference training was not run and its reported scores were not independently validated. No reference image, GIF, result number or source block is copied into this repository.

## Frozen local sources

These byte-level SHA-256 values identify files actually read on 2026-10-25, rather than an authenticated upstream commit. No licence file is present in this reference directory; permission to transplant code has not been established. Confirm the applicable licence before future code reuse.

| Local file | SHA-256 |
| --- | --- |
| `modules.py` | `9fa1d62c00e61a305a2a18907c5250793fccbcf0250c7752a8124800cef481d9` |
| `dataset.py` | `5db7b5147fa4e7daaac3c262079a6701bf7d73d7fac9fb792c75f46a59a76e67` |
| `train.py` | `4e1281e579b11f2072f26d9821eb19dac580502a95c6f07ad96e51604f4ec5b0` |
| `predict.py` | `1aca71413ddf046f7479679ad730a354bb440afc985a647fa36320de60377ae5` |
| `utils.py` | `e77a9788fd236ddcdb5f6b8290ce61f627e68b200833787200d7fd1a8408dfae` |
| `README.MD` | `0a53482f16743583ad0847f5bd7905bed20b18fa788738958e11690571711860` |

## Tensor flow and model identity

In `modules.py`, `ContextModule.context` applies IN → LeakyReLU → 3³ convolution → Dropout3d(0.3) → IN → LeakyReLU → 3³ convolution. `DecreaseLayer` first applies convolution/activation and adds the context output to that intermediate tensor. Default encoder channels are 16, 32, 64, 128, 256: five resolutions and four stride-2 reductions. For a divisible spatial extent S, the scales are S, S/2, S/4, S/8 and S/16. The bottleneck upsamples to 128 channels, concatenates a 128-channel skip to form 256 and localisation returns 128. Later localisation joins 64+64 and returns 64, then joins 32+32 and returns 32; the final join is 16+16 before a 32-channel final convolution. This differs from our three-reduction `b, 2b, 4b, 8b` skeleton.

`UpsamplingModule` uses trilinear interpolation (`align_corners=True`), convolution and activation. `LocalisationModule` uses 3³ then 1³ convolutions with activations. Three six-class segmentation maps are combined: upsample the low map and add it to the middle map, then upsample that sum and add the final map. Head widths are hard-coded (64, 32, 32) and the initial width is fixed at 16; an exposed `features` argument does not guarantee arbitrary-width support. The reference does not establish checkerboard artefacts in our historical implementation or prove that interpolation repairs a measured defect.

Our primary Improved is `legacy_improved_v1`: residual blocks, three transposed-convolution decoder expansions and auxiliary training losses weighted 0.25 each; evaluation uses final-head logits only. Separate `improved_context_v2` has dropout, interpolation, inference-time fusion of resized auxiliary logits and auxiliary training supervision. Shared paper motivation does not make these structures or checkpoints interchangeable. Our model factory selects saved configuration; new Improved training defaults to Context v2, while historical reproduction explicitly selects legacy.

## Probability, data and metric contracts

The reference `ImprovedUNet.forward` ends in `Softmax(dim=1)`. `SoftDiceLoss.forward` consumes those probabilities and one-hot targets and returns negative mean soft Dice over samples/classes, without CE. Our models return logits for weighted cross-entropy plus weighted soft Dice, with integer targets for CE and softmax inside Dice/probability computation. Passing already normalised reference outputs directly into our CE changes its intended objective. Auxiliary logit fusion and auxiliary loss supervision are different operations.

Reference `create_loader` loads arrays and applies TorchIO transforms to every image once before constructing `TensorDataset`. `ScalarImage` and `LabelMap` distinguish intensities from labels and share a subject for geometry transforms. This is precomputed augmentation at loader creation, not fresh sampling every epoch. `load_prostate_data` passes `transform_flag` to both train and validation; `train_model` sets it True, so this copy also precomputes augmented validation data. Test explicitly passes False. Our augmentation is sampled by the training dataset after cropping, and validation/test reject augmentation. Learn paired geometry handling without copying this validation policy.

The reference splits sorted image/label filename lists by fractional index; it does not verify patient separation. Our frozen manifest enforces 27/6/5 patients and authoritative label provenance. Reference `dice_score` sums batch and spatial axes per class; `evaluate_unet` averages batch scores equally and then averages six classes. Our headline computes prostate hard Dice per volume, averages volumes within each patient, then weights the five patients equally. Reference per-class scores cannot be ranked against our patient-prostate headline.

## Training, presentation and applied decisions

Reference constants are 30 epochs, Adam LR 4e-4 and cosine minimum 1e-4. `scheduler.step()` follows each epoch's training, validation and LR logging. The state is saved after the loop; that path has no validation-best checkpoint save. Our primary experiment selects highest `validation_mean_dice`; later Context uses mean patient prostate Dice. Retain explicit best-checkpoint/configuration identity. Context cosine arm C underperformed B, so a borrowed schedule is not improvement evidence.

Reference visualisation reads the first batch of a shuffled validation loader; its case is not guaranteed fixed across epochs. Our lexicographically fixed validation case makes comparisons more interpretable, and static summaries keep the main README readable. Distinguish spatial animation frames from checkpoint epochs. A favourable slice does not replace patient metrics, resource gates or held-out failure analysis. We do not reproduce the reference's claim of fully implementing the paper or borrow its experience statements.

The applied learning is clearer module order, tensor flow, loss interface, scheduler timing and visual selection documentation. No new network, copied asset, training or held-out inference is part of this record. Future reuse requires licence confirmation, source identity, a change record and regression evidence. The author should be able to redraw both tensor flows and explain augmentation timing, train/validation/test roles, fusion versus supervision and why extra complexity has not established a stable current-task advantage. AI disclosure remains for the author.
