# HipMRI label-mapping evidence

This directory freezes the evidence used to interpret the indexed NIfTI
semantic masks. It is not a model result and does not authorise a clinical
use of the model.

## Confirmed mapping

`prostate_label = 5`

Authoritative source: `sources/csiro-brief-data-description.txt`, downloaded
from the supporting-file endpoint of the CSIRO collection *Labelled weekly MR
images of the male pelvis*, version 2, DOI
[`10.25919/45T8-P065`](https://doi.org/10.25919/45T8-P065).

The source explicitly states:

```text
0=Background
1=Body
2=Bones
3=Bladder
4=Rectum
5=Prostate
```

The mapping appears under `Labels:` near the end of that file. Its SHA-256 is
`DC3A99204684CED859872D79A75707AF3C0AD31D00F5F3213063CE5E30727EE6`.

## Downloaded evidence

| Local file | What it establishes | Source and locator | SHA-256 |
| --- | --- | --- | --- |
| `sources/csiro-brief-data-description.txt` | The authoritative numeric mapping for the released indexed masks. | CSIRO supporting file, collection v2, DOI `10.25919/45T8-P065`; `Labels:` block. | `DC3A99204684CED859872D79A75707AF3C0AD31D00F5F3213063CE5E30727EE6` |
| `sources/can3d-compact-context-aggregation-arxiv-2109.05443.pdf` | Independent study context: the release contains 211 images and six classes. The paper numbers those classes from 1 to 6, so its prostate class 6 corresponds to this NIfTI file's zero-based value 5, as confirmed directly by the CSIRO mapping above. | Dai et al., *CAN3D*, arXiv:2109.05443, PDF p. 8. | `2583AEDFA2233E101FA0E96F45A818AC7D930418F638D963EC78E1DDE5664964` |
| `sources/dowling-2015-pubmed-medline-26581150.txt` | Bibliographic record and abstract for the source study cited by CSIRO; it confirms prostate, rectum, bladder and bones were manually contoured. It does **not** define the released numeric mapping. | NCBI E-utilities MEDLINE record PMID `26581150`; DOI `10.1016/j.ijrobp.2015.08.045`. | `95FFAD007CC9D01F7F23CC9F7A3B7E17281B86E40766F088E0814F2EBB6C02CA` |

## Required experiment wording

Use this exact evidence identity in formal run metadata:

```text
prostate_label: 5
prostate_label_source: CSIRO, Labelled weekly MR images of the male pelvis,
v2, DOI: 10.25919/45T8-P065, BriefDataDescription.txt, Labels block:
5=Prostate.
```

Any future replacement source must be an authoritative source that explicitly
maps a numeric value to prostate. A source merely stating that prostate was
segmented is supporting context, not a substitute for the mapping.

## Architecture references from the Hard-difficulty brief

The following papers are implementation background only. They do not change
the HipMRI data split, label mapping, evaluation protocol, or acceptance
criterion.

| Course reference | Local file | Verified identity and relevance | SHA-256 |
| --- | --- | --- | --- |
| `[2]` | `sources/isensee-2018-brats-2017-challenge-arxiv-1802.10508.pdf` | Isensee et al., *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*, arXiv:1802.10508; the open preprint corresponding to the 2018 BrainLes chapter, DOI `10.1007/978-3-319-75238-9_25`. It is background for residual and deep-supervision design decisions, not HipMRI performance evidence. | `91E206BDC48D09B3BD5204192EE39780B6DD0027C60DAA66FC5524E0E94DF78A` |
| `[3]` | `sources/yu-koltun-2016-multi-scale-context-aggregation-arxiv-1511.07122.pdf` | Yu and Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016, arXiv:1511.07122. It is background for dilated-convolution context aggregation. | `ACAC9B267CF8B2498D53FE9A9A5AFCC51B3DC1CC677AE0AE82F8CA626869A8B1` |
| `[4]` | `sources/cicek-2016-3d-unet-arxiv-1606.06650.pdf` | Çiçek et al., *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation*, MICCAI 2016, arXiv:1606.06650. It is the volumetric U-Net baseline reference. | `267BA70094E23CE2655481EEC5BCA7CEA98EB4797390DAA94476196A1AA31067` |
| `[5]` | `sources/can3d-compact-context-aggregation-arxiv-2109.05443.pdf` | Dai et al., *CAN3D: Fast 3D Medical Image Segmentation via Compact Context Aggregation*, arXiv:2109.05443 (2021). This is the locally stored public preprint corresponding to the abbreviated 2021 context-aggregation citation; the later journal version has DOI `10.1016/j.media.2022.102562`. | `2583AEDFA2233E101FA0E96F45A818AC7D930418F638D963EC78E1DDE5664964` |
