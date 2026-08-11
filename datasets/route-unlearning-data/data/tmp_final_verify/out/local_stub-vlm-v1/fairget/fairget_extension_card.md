# fairget CelebA-40 extension card

- Benchmark: `fairget`
- Source version: `fairget-2026.07`
- Prompt registry hash: `642c7cc2ff4226aa`
- Annotator model fingerprint: `0bc9c9174b68d756`

## Counts
- annotation_rows: 84
- qa_eval: 0
- qa_train: 0
- route_probes: 4
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
