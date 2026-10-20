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
