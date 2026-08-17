# fiubench CelebA-40 extension card

- Benchmark: `fiubench`
- Source version: `fiubench-8e12cdd`
- Prompt registry hash: `28481d2ebb85a6fb`
- Annotator model fingerprint: `de9856e1d2620d4e`

## Counts
- annotation_rows: 560000
- qa_eval: 16240
- qa_train: 77420
- route_probes: 500
- splits: 3

## Label provenance
All CelebA-style predictions are **weak labels** produced by a frozen
protocol and stored under `extended_attributes.celeba40.*`. Source
annotations (e.g. `source_attributes.fairface.*`) are never overwritten.
Observations carry one of three tiers: high-confidence automatic,
human-verified, or unlabeled/uncertain (`label: null`).

## Sensitive and low-reliability labels
- Subjective/sensitive (inherit CelebA definitions and limitations): ['Attractive', 'Male', 'Young']
- Low-reliability source labels: ['High_Cheekbones', 'Oval_Face', 'Pointy_Nose']
`Male` is the CelebA binary annotation, not a person's self-identified
gender. These labels must not be interpreted as ground-truth identity.

## Images
Images are referenced by URI/SHA-256 only and are **not** redistributed.

## License and citation
- License: See the upstream dataset license.
