Minimum Viable Research Plan: Testing Direct and Identity-Mediated Knowledge Routes in an MLLM

Status: Research MVP specificationLast source check: 2026-08-04Primary question: Does an MLLM trained with direct image-to-attribute supervision learn a different computational route from an MLLM trained to compose image-to-identity and identity-to-attribute mappings?

1. Scope

This project is a minimum viable research experiment, not an engineering product.

The MVP should answer one narrow question:

Given the same face images and the same final property labels, do different supervision structures produce different behavioral dependencies, internal information-flow patterns, and parameter updates?

The two hypothesized routes are:

[X_p \rightarrow A_{\text{visual}} \rightarrow Y]

and

[X_p \rightarrow I_p \rightarrow K(I_p,A) \rightarrow Y,]

where:

(X_p) is an image of person (p);

(A_{\text{visual}}) is directly visible evidence for a property;

(I_p) is the model's internal identity representation;

(K(I_p,A)) is an identity–property association;

(Y) is the generated property answer.

In scope

Ten CelebA identities with repeated images.

Synthetic aliases and synthetic binary properties.

A synthetic visual marker that provides a direct perceptual solution.

Three required training conditions and one optional mixed condition.

LoRA/QLoRA fine-tuning of one MLLM.

Behavioral conflict tests.

Attention-edge knockout and causal activation patching.

Layer-wise linear probes.

LoRA update-distribution analysis.

Three optimization seeds.

Explicitly out of scope

A general-purpose training framework.

A web interface, database, service, API, authentication, or deployment pipeline.

Distributed multi-node training.

Automated hyperparameter optimization.

Production monitoring.

Model unlearning in the first experiment.

Claims about human gender or sensitive personal attributes.

Publishing or redistributing CelebA images or derived image files.

CelebA is available for non-commercial research and its official agreement restricts copying and redistribution. Keep the images local and release only code, manifests containing non-image metadata, aggregate results, and instructions. See the official CelebA page.

2. Research hypotheses

H1: Direct supervision creates cue-dependent behavior

A model trained on:

User: [IMAGE]
What property is shown?
Assistant: DAX

should primarily learn:

[X_{\text{marker}} \rightarrow Y.]

Expected behavior:

generalizes to unseen identities carrying the same marker;

follows the marker when marker and identity association conflict;

remains accurate when the face is obscured but the marker is visible;

degrades when the marker is removed;

shows strong causal dependence on marker-region image tokens.

H2: Separated mediated supervision creates identity-dependent composition

A model trained on two disjoint example types:

User: [IMAGE]
Who is this?
Assistant: Vela_07

and:

User: What property does Vela_07 have?
Assistant: DAX

should learn:

[X_p \rightarrow I_p\quad\text{and}\quadI_p \rightarrow K(I_p,A) \rightarrow Y.]

The image and target property must never occur in the same training context in this condition.

Expected behavior:

answers properties for held-out images of known identities;

fails or weakens on unseen identities;

follows the identity-associated property when the marker conflicts;

degrades when the face is obscured;

depends more strongly on face-to-query or identity-state transfer than the direct model.

H3: An explicit textual sequence does not guarantee mediation

A model trained on:

User: [IMAGE]
Identify this person and state their property.
Assistant: This is Vela_07. Vela_07 has DAX.

may still use the image directly to predict DAX.

The printed chain:

image -> identity text -> property text

is not evidence that the internal causal path follows the same chain.

H4: Different routes produce different update distributions

Relative to a common base checkpoint, the direct and mediated adapters should differ in at least one of:

layer-wise update energy;

attention-versus-MLP update allocation;

query/key/value/output update allocation;

singular-value spectra;

layer-wise task-vector cosine similarity;

causal sensitivity to disabling adapters in individual layers.

Parameter differences alone will not prove different routes. They must be interpreted with behavioral and causal-intervention evidence.

3. Why use synthetic properties and aliases?

Use CelebA only as a source of repeated face images and identity labels.

Do not begin with natural attributes such as Male, Young, or Attractive. Natural attributes introduce:

annotation noise;

ambiguity;

correlations with face identity;

correlations with pose, background, lighting, and image source;

possible pretraining knowledge;

ethical and interpretive problems.

CelebA contains 202,599 images, 10,177 identities, and 40 binary attributes, according to the official dataset page. An audit of CelebA annotations reported substantial inconsistency for several attributes, which further supports using synthetic labels in the first study: Wu et al., CVPRW 2023.

Synthetic identity aliases

Map the selected integer identities to arbitrary aliases:

Vela_01
Vela_02
...
Vela_10

Do not use celebrity names. The base MLLM might already recognize a celebrity or know facts about them, creating an uncontrolled pretrained route.

Synthetic binary properties

Assign five aliases to DAX and five to WUG.

These labels should be:

semantically meaningless;

tokenizable without excessive fragmentation;

consistently formatted;

checked against the base model to ensure no strong prior preference.

Before training, measure:

P(DAX | image, question)
P(WUG | image, question)
P(DAX | alias)
P(WUG | alias)

The base model should be near indifferent after restricting evaluation to the two candidate-token logits.

4. Direct visual cue

Overlay a small synthetic marker on each face image:

DAX: blue square;

WUG: orange triangle.

The marker should:

have a fixed approximate size;

appear in a fixed corner or a small set of controlled locations;

not overlap the face;

be large enough to cover several visual tokens;

have a known bounding box;

be applied dynamically with Pillow instead of saving redistributed derived images.

This creates two independently controllable sources of the answer:

[\text{marker pixels} \rightarrow \texttt{DAX/WUG}]

and:

[\text{face} \rightarrow \texttt{Vela_k}\rightarrow \texttt{DAX/WUG}.]

Required image variants

For every selected image, create the following at runtime:

Variant

Face

Marker

Purpose

aligned

original

identity-consistent

normal task

conflict

original

opposite property

route preference

no_marker

original

absent

identity route sufficiency

face_masked

blurred/covered

present

direct route sufficiency

face_masked_no_marker

blurred/covered

absent

chance/control

neutral_marker

original

neutral symbol

marker-specific control

Keep the image preprocessing identical across training conditions.

5. Dataset selection

Selection rule

Parse CelebA's identity annotation file.

Count images per identity.

Keep identities with at least 20 images.

Fix a selection seed, for example 20260804.

Randomly sample exactly ten eligible identities.

Freeze the selection in manifests/identity_manifest.json.

Assign aliases with no semantic relation to the identity.

Assign five identities to each synthetic property using a second fixed seed.

Never resample based on model performance.

A threshold of 20 images is preferable to the original “more than 10” criterion because it supports meaningful train/validation/test splits.

Split

For each identity:

Split

Images

Total for 10 identities

Train

12

120

Validation

4

40

Test

4

40

All conditions use the same split.

Manifest schema

{
  "image_file": "000001.jpg",
  "celeba_identity_id": 1234,
  "alias": "Vela_07",
  "property": "DAX",
  "split": "train",
  "face_bbox": [10, 10, 160, 180],
  "marker_bbox": [8, 8, 48, 48]
}

If face boxes are not available from the chosen CelebA files, use the aligned/cropped images and define a fixed conservative face region. For the MVP, a perfect segmentation mask is unnecessary.

6. Training conditions

Train every adapter from the same immutable base checkpoint.

C0: Base model

No fine-tuning.

Use it to check:

pretrained identity leakage;

prior preference for aliases;

prior preference for DAX/WUG;

marker sensitivity before training.

C1: Direct

Training example:

User: [ALIGNED IMAGE]
What property is shown?
Assistant: DAX

Constraints:

no alias in the prompt or answer;

property is determined by the marker;

identity-property assignment is aligned during training.

C2: Joint sequence

Training example:

User: [ALIGNED IMAGE]
Identify this person and state their property.
Assistant: This is Vela_07. Vela_07 has DAX.

This is the original sequence design.

Its scientific role is to test whether an explicit intermediate identity output causes an identity-mediated internal route or permits a direct shortcut.

C3: Separated mediated

Use two disjoint datasets.

C3a: Image-to-identity

User: [IMAGE]
Who is this?
Assistant: Vela_07

For these examples:

use no marker, or randomize DAX/WUG markers independently of identity;

never include the property in the prompt or answer.

C3b: Alias-to-property

User: What property does Vela_07 have?
Assistant: DAX

For these examples:

use text only;

never include an image.

The mediated adapter is trained on a mixture of C3a and C3b examples, but no example contains both an image and its target property.

C4: Mixed, optional but valuable

Mix:

direct image-to-property examples;

image-to-identity examples;

alias-to-property examples.

This condition tests:

coexistence of multiple routes;

preferred route under aligned input;

route choice under conflict;

fallback behavior after knocking out one path.

Exposure matching

Exact matching is difficult because C3 contains two tasks. Record and approximately match:

optimizer updates;

image presentations;

alias-token exposures;

property-token exposures;

supervised answer-token count;

number of examples per identity.

Save these counts with every training run.

7. Model choice

Primary model

Use:

Qwen/Qwen3-VL-4B-Instruct

Rationale:

it is the principal model in the Pathways of Visual Information Flow in Vision-Language Models repository;

the public Pathways implementation already handles Qwen3-VL image-token ranges, causal patching, and attention knockout;

Hugging Face Transformers provides a dedicated Qwen3-VL implementation and processor.

The Pathways repository also supports LLaVA-1.5-7B and InternVL3.5-4B, but use only one model for the MVP. Cross-model replication belongs after the first experiment succeeds.

8. Source code and libraries to build on

8.1 Primary base: israfelsr/vlm-pathways

Repository: israfelsr/vlm-pathways

License: MIT

Paper: Pathways of Visual Information Flow in Vision-Language Models

The repository is the main base for the mechanistic portion. Its README documents Qwen3-VL, LLaVA, and InternVL configs; Hugging Face on-disk datasets; baseline evaluation; attention knockout; paired causal patching; corrupted patching; source-dominance analysis; and text recovery.

Reuse the following components.

vlm_spatial/model.py

Reuse:

model architecture detection;

AutoProcessor;

eager attention loading;

device placement.

Modify or wrap it for training because the original loader puts the model in evaluation mode.

vlm_spatial/data.py

Reuse:

Hugging Face dataset loading;

question construction patterns;

image/text token-range detection.

Extend it with:

def find_face_patch_indices(...): ...
def find_marker_patch_indices(...): ...

vlm_spatial/hooks.py

Reuse:

language-layer discovery;

pre-softmax attention-mask blocking;

attention boosting;

direct last-token-to-image knockout;

text-token-to-image knockout;

last-token-to-text knockout.

The repository uses PyTorch forward pre-hooks to modify attention masks. PyTorch officially supports input/output intervention through register_forward_pre_hook and register_forward_hook; see the PyTorch module hook documentation.

experiments/causal_tracing.py

Reuse:

hidden-state collection;

layer-output replacement;

paired clean/corrupted activation patching;

probability/logit recovery measurement;

image-grid and image-patch-region utilities.

Refactor only the minimum needed to support person, marker, and identity-token groups.

License obligation

The Pathways repository uses the MIT license. If code is copied or substantially modified, retain its copyright and license notice.

8.2 Hugging Face Transformers

Use for:

Qwen3-VL model and processor loading;

chat-template rendering;

multimodal input construction;

generation and logits;

model configuration.

Resources:

Qwen3-VL model documentation

Transformers repository

8.3 Hugging Face PEFT

Use for:

LoRA adapter injection;

selecting target modules;

saving/loading adapters;

obtaining merged adapter deltas for parameter analysis.

Resources:

PEFT documentation

PEFT LoRA guide

Use the same LoRA initialization seed in all conditions.

8.4 Hugging Face Accelerate

Use for a small custom training loop with:

device placement;

BF16 or FP16 mixed precision;

gradient accumulation;

reproducible single-GPU execution.

Resources:

Accelerate overview

Gradient accumulation

Do not add DeepSpeed or FSDP unless the model cannot be trained on available hardware. They are unnecessary complexity for the MVP.

8.5 Hugging Face Datasets

Use for:

local manifests;

PIL-backed image columns;

train/validation/test splits;

saving small processed metadata datasets to disk;

compatibility with the Pathways repository's .hf format.

Resources:

Load image data

Create an image dataset

Do not push the CelebA-derived dataset to the Hub.

8.6 bitsandbytes, optional

Use only if the 4B model does not fit for LoRA training in BF16.

Transformers integrates bitsandbytes for 8-bit and 4-bit model loading. The documented 4-bit workflow supports training extra parameters, which fits QLoRA adapter training:

Transformers bitsandbytes documentation

Use a BF16 base when feasible because quantization can complicate parameter and representation comparisons.

8.7 PyTorch

Use directly for:

training;

hooks;

activation capture;

activation replacement;

attention-mask intervention;

tensor norms and SVD;

deterministic seeding.

Resources:

PyTorch modules and hooks

Module hook API

8.8 scikit-learn

Use only for lightweight analysis:

StandardScaler;

multinomial or one-vs-rest LogisticRegression identity probes;

binary LogisticRegression property probes;

stratified cross-validation;

PCA for visualization, not as primary evidence.

Resources:

Linear models

StandardScaler

PCA

8.9 SciPy

Use for:

scipy.linalg.svdvals;

simple statistical tests or confidence-interval utilities if needed.

Resource:

scipy.linalg.svdvals

PyTorch's torch.linalg.svdvals is also sufficient. Do not introduce SciPy only for SVD if it is not otherwise needed.

8.10 Safetensors

Use for:

loading saved LoRA adapter weights;

storing extracted activation summaries or merged deltas when appropriate.

Resources:

Safetensors overview

Safetensors Torch API

8.11 Small utility libraries

Use:

Pillow: marker overlay and face masking;

NumPy: arrays and aggregate metrics;

pandas: result tables;

matplotlib: figures;

PyYAML: experiment configuration;

tqdm: progress bars;

pytest, optional: only a few data/hook smoke tests.

Do not add a heavy experiment-management dependency for the MVP. CSV/JSONL outputs and a run directory are sufficient.

9. Environment

The Pathways repository pins the following core environment for its paper code:

torch==2.8.0
torchvision==0.23.0
transformers==4.57.0
accelerate==1.10.1
datasets==4.2.0
safetensors==0.6.2
sentencepiece==0.2.1
einops==0.8.1
numpy==2.2.6
pillow==11.3.0
matplotlib==3.10.7
pyyaml==6.0.3
tqdm==4.67.1

Add:

peft
pandas
scikit-learn
scipy

Optional:

bitsandbytes
pytest

Recommended setup

git clone https://github.com/israfelsr/vlm-pathways.git
cd vlm-pathways

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install peft pandas scikit-learn scipy

If QLoRA is required:

pip install bitsandbytes

Create a fresh lock file after confirming that Qwen3-VL loads correctly. Record:

python --version
pip freeze
nvidia-smi

The Pathways package declares Python 3.10 or newer.

10. Minimal project structure

Fork or clone vlm-pathways, then add a small research subdirectory:

vlm-pathways/
├── configs/
│   ├── route_direct.yaml
│   ├── route_joint.yaml
│   ├── route_mediated.yaml
│   └── route_mixed.yaml
├── data/
│   └── celeba_route_mvp/
│       ├── manifests/
│       │   ├── identity_manifest.json
│       │   ├── train.jsonl
│       │   ├── validation.jsonl
│       │   └── test.jsonl
│       └── README.md
├── experiments/
│   ├── route_prepare_celeba.py
│   ├── route_train.py
│   ├── route_evaluate_behavior.py
│   ├── route_evaluate_pathways.py
│   ├── route_extract_activations.py
│   ├── route_train_probes.py
│   └── route_analyze_adapters.py
├── vlm_spatial/
│   ├── regions.py
│   └── route_dataset.py
└── results/
    └── route_mvp/
        ├── run_manifest.json
        ├── adapters/
        ├── behavior/
        ├── pathways/
        ├── probes/
        ├── parameters/
        └── figures/

Do not create a separate reusable Python package unless modifications to vlm_spatial become substantial.

11. Script plan

11.1 route_prepare_celeba.py

Responsibilities:

Read CelebA identity annotations.

Count images per identity.

Sample ten eligible identities.

Assign aliases and properties.

Create deterministic image splits.

Write JSON/JSONL manifests.

Verify no image crosses splits.

Print per-identity counts.

Example:

python experiments/route_prepare_celeba.py \
  --celeba-root /local/path/to/celeba \
  --min-images 20 \
  --n-identities 10 \
  --train-per-id 12 \
  --val-per-id 4 \
  --test-per-id 4 \
  --seed 20260804 \
  --output data/celeba_route_mvp/manifests

Output only metadata. Do not copy images.

11.2 vlm_spatial/route_dataset.py

Implement one small PyTorch/Hugging Face-compatible dataset class.

Responsibilities:

load an image path from the local CelebA root;

overlay aligned, conflicting, random, neutral, or no marker;

mask or blur the face when requested;

format the prompt for the selected condition;

format assistant labels;

return metadata such as face and marker boxes.

Suggested pre-tokenization record:

{
    "image": pil_image,
    "question": str,
    "answer": str,
    "identity": int,
    "alias": str,
    "property": str,
    "variant": str,
    "face_bbox": tuple,
    "marker_bbox": tuple,
}

11.3 route_train.py

Implement a small custom training loop instead of a general trainer abstraction.

Responsibilities:

load Qwen3-VL and its processor;

optionally load in 4-bit;

freeze the base model;

inject LoRA;

construct labels masking prompt tokens;

train one condition and one seed;

save adapter, config, losses, and exposure counts.

Pseudocode:

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=config.grad_accumulation_steps,
)

model = load_qwen3_vl(config)
model = get_peft_model(model, lora_config)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)

for batch in train_loader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

Do not add callbacks, plugin registries, or a generic configuration framework.

11.4 route_evaluate_behavior.py

For each adapter, seed, test image, and input variant:

run one forward pass;

store logits for DAX and WUG;

store greedy answer;

calculate identity prediction when relevant;

store prompt, alias condition, marker condition, and masking condition.

Write one JSONL or Parquet row per evaluation example.

11.5 vlm_spatial/regions.py

Move or adapt image-grid code from experiments/causal_tracing.py.

Implement:

def get_image_grid(inputs, n_image_tokens): ...

def bbox_to_image_token_indices(
    image_range,
    grid_hw,
    image_size,
    bbox,
    padding=0,
): ...

Return absolute sequence token indices for marker, face, and background regions.

11.6 route_evaluate_pathways.py

Adapt the repository's attention hooks.

Required interventions:

final token cannot attend to marker image tokens;

final token cannot attend to face image tokens;

query tokens cannot attend to marker image tokens;

query tokens cannot attend to face image tokens;

final token cannot attend to query text;

all image tokens blocked from final token;

all image tokens blocked from query tokens.

Run interventions over:

all layers;

early third;

middle third;

late third.

Only add fine-grained layer sweeps after the coarse sweep identifies relevant regions.

11.7 route_extract_activations.py

Use forward hooks to extract hidden states at every fourth decoder layer for:

pooled face image tokens;

pooled marker image tokens;

pooled all-image tokens;

pooled question tokens;

final prompt token.

Store reduced summaries rather than every token activation if disk usage becomes large.

11.8 route_train_probes.py

For each model, layer, and token group:

standardize activations using training activations only;

fit identity logistic regression;

fit property logistic regression;

evaluate on held-out images;

report balanced accuracy and cross-entropy;

repeat with at least three train/test resampling seeds.

Prevent image leakage: keep all representations from one image in the same fold.

11.9 route_analyze_adapters.py

Load LoRA adapter tensors and compute:

merged update (\Delta W=sBA);

Frobenius norm;

update-to-base norm ratio;

update energy per layer;

SVD singular values;

effective rank;

pairwise task-vector cosine similarity;

layer-wise similarity;

attention-versus-MLP energy;

Q/K/V/O and MLP projection energy.

Also implement functional adapter ablation:

temporarily disable the LoRA contribution in one layer;

rerun aligned and conflict evaluation;

record behavioral changes.

12. Example configuration

experiment_name: route_direct
condition: direct
seed: 0

model:
  name: Qwen/Qwen3-VL-4B-Instruct
  dtype: bfloat16
  quantization: null
  gradient_checkpointing: true
  freeze_vision_encoder: true

data:
  celeba_root: /local/path/to/celeba
  manifest_dir: data/celeba_route_mvp/manifests
  image_size: 448
  marker_size_px: 40
  marker_location: top_left
  train_variant: aligned

training:
  epochs: 10
  learning_rate: 0.0001
  weight_decay: 0.0
  warmup_ratio: 0.05
  batch_size: 1
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0
  evaluation_every_steps: 20

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

generation:
  max_new_tokens: 8
  do_sample: false

Treat these values as starting points. Use one development run to select the learning rate and number of updates, then freeze the choices for every condition and seed.

13. Behavioral evaluation

Competence tests

The mediated route is uninterpretable unless both component mappings work.

Measure:

[\operatorname{Acc}(I_p\mid X_p),]

[\operatorname{Acc}(A\mid I_p),]

and composed performance:

[\operatorname{Acc}(A\mid X_p).]

Conflict test

For a person whose stored property is DAX, overlay the WUG marker.

Compute restricted candidate logits:

\operatorname{logit}(\texttt{WUG})

\operatorname{logit}(\texttt{DAX}).]

Positive values indicate direct-marker preference; negative values indicate identity-association preference.

Marker dependence

\operatorname{Acc}_{\text{aligned}}

\operatorname{Acc}_{\text{no-marker}}.]

Face dependence

\operatorname{Acc}_{\text{aligned}}

\operatorname{Acc}_{\text{face-masked}}.]

Unseen-identity generalization

Reserve a small additional set of CelebA identities never used during fine-tuning. Add DAX/WUG markers and test image-to-property prediction.

Expected:

Direct: above chance and similar to known identities.

Mediated: near chance unless it learns an unintended direct marker mapping.

Joint: empirical result.

Mixed: likely above chance.

Alias intervention

Use:

no alias;

correct alias;

wrong alias;

unrelated alias.

This tests whether textual identity information changes the property answer when pixels are fixed.

14. Mechanistic analysis

14.1 Causal patching

Create clean/counterfactual pairs:

same face, opposite marker;

different face, same marker;

same face, correct versus incorrect alias;

same identity, aligned versus conflict input.

At layer (l), copy the hidden state of token group (G) from the clean run into the counterfactual run.

Measure:

P(y_{\text{clean}}\mid\text{patched})

P(y_{\text{clean}}\mid\text{counterfactual}).]

Token groups:

marker patches;

face patches;

all image tokens;

question text;

alias text;

final prompt token.

Expected direct signature:

high restoration from marker patches;

low restoration from face or alias representations.

Expected mediated signature:

restoration from face tokens at identity-forming layers;

later restoration from alias/query/final-token states.

14.2 Attention-edge knockout

The Pathways code blocks selected query–key pairs by adding a large negative value to the pre-softmax attention mask.

Use:

last -> marker image tokens;

last -> face image tokens;

query text -> marker image tokens;

query text -> face image tokens;

last -> alias/query text.

Interpretation:

patching asks which information is normally carried;

knockout asks whether a route is necessary under intervention.

Do not infer route use from attention weights alone.

14.3 Probe analysis

Fit identity and property probes by layer.

A route-ordering pattern would be informative:

direct: property decodable from marker/image states before identity is strongly decodable;

mediated: identity decodable before property becomes available in downstream text/final-token states.

Probe evidence remains correlational. Require agreement with patching or knockout evidence.

15. Parameter analysis

Let the base parameter matrix be (W_0), and the merged LoRA-adapted matrix be:

[W_c = W_0 + \Delta W_c.]

Analyze (\Delta W_c), not raw (W_c).

Layer-wise relative update

[r_m =\frac{\lVert\Delta W_m\rVert_F}{\lVert W_{m,0}\rVert_F+\epsilon}.]

Layer update energy

[E_l =\frac{\sum_{m\in l}\lVert\Delta W_m\rVert_F^2}{\sum_m\lVert\Delta W_m\rVert_F^2}.]

Effective rank

For singular values (\sigma_i), define:

[p_i = \frac{\sigma_i}{\sum_j \sigma_j},\qquadr_{\text{eff}} = \exp\left(-\sum_i p_i\log p_i\right).]

Task-vector similarity

\frac{\Delta\theta_a^\top\Delta\theta_b}{\lVert\Delta\theta_a\rVert\lVert\Delta\theta_b\rVert}.]

Report global, layer-wise, and module-family similarity.

Functional adapter ablation

For each layer (l):

disable LoRA contribution in layer (l);

rerun aligned and conflict tests;

measure change in property logit and accuracy.

This provides stronger evidence than update magnitude alone.

16. Minimum experiment matrix

Required:

Condition

Seeds

Adapters

Direct

3

3

Joint sequence

3

3

Separated mediated

3

3

Optional:

Condition

Seeds

Adapters

Mixed

3

3

Total required: 9 adapters. Total with Mixed: 12 adapters.

Use the same selected identities and image splits across all runs.

17. Statistical analysis

The unit of generalization is not simply the number of images because images from the same identity are correlated.

For the MVP:

report per-identity metrics;

aggregate first within identity, then across identities;

report means and bootstrap confidence intervals over identities;

show all three optimization seeds;

do not rely only on a single aggregate accuracy.

For conflict scores, present identity-level means, paired condition differences, and bootstrap intervals. Avoid overcomplicating the statistics for a ten-identity pilot.

18. Minimum success criteria

Basic competence

Direct aligned property accuracy: at least 80%.

Mediated image-to-identity accuracy: at least 80%.

Mediated alias-to-property accuracy: at least 95%.

Mediated image-to-property composition: clearly above 50%.

Behavioral route separation

Direct follows the visual marker in conflicts substantially more than Mediated.

Mediated follows the identity association substantially more than Direct.

Direct generalizes to unseen identities carrying the marker.

Mediated is more dependent on face visibility.

A practical pilot target is at least a 30-percentage-point condition difference in conflict preference, but report effect sizes rather than treating this as a universal threshold.

Mechanistic separation

At least two of:

Direct is more affected by marker-to-final knockout.

Mediated is more affected by face-to-query knockout.

Direct patching restoration is concentrated in marker image tokens.

Mediated restoration proceeds from face/identity representations to later text/final-token states.

Identity becomes decodable earlier than property in the mediated model.

The adapters depend on different layers under functional adapter ablation.

Parameter separation

At least one reproducible difference in:

update-energy distribution;

module-family allocation;

effective rank;

layer-wise task-vector similarity;

functional adapter-layer importance.

Behavioral and causal evidence are primary. Parameter separation is supporting evidence.

19. Failure cases and interpretation

Mediated condition cannot compose image to property

Possible causes:

image-to-identity accuracy is too low;

identity representation appears too late to activate factual recall;

aliases tokenize poorly;

the model does not learn multi-hop composition from the small dataset.

Response:

improve image-to-identity training first;

increase alias-to-property examples;

add a test prompt that explicitly asks the model to identify first;

do not add direct image-property examples to the mediated condition.

Joint sequence behaves like Direct

This is a meaningful result:

Producing an identity token before a property token does not ensure that the property is causally retrieved through identity.

All conditions follow the marker

Potential causes:

the marker is too salient;

mediated image-to-identity examples accidentally correlate marker with property;

base model or preprocessing creates a marker shortcut.

Response:

remove markers entirely from mediated identity training;

use randomized markers;

reduce marker size.

All conditions follow identity

Potential causes:

marker too small or poorly represented;

visual encoder loses marker detail;

direct training insufficient;

property tokens associated mainly with identities.

Response:

enlarge marker;

move marker into a well-tokenized region;

verify direct training on unseen identities.

Knockout has little effect

Potential causes:

alternate route compensates;

layer range is wrong;

image-token indices are wrong;

attention implementation does not expose the expected mask;

output is supported by residual information written earlier.

Response:

run causal patching;

verify hook call counters;

compare early/middle/late layer ranges;

inspect token ranges;

use corrupted-input recovery.

20. Reproducibility checklist

For every run, save:

{
  "git_commit": "...",
  "base_model": "Qwen/Qwen3-VL-4B-Instruct",
  "base_model_revision": "...",
  "condition": "direct",
  "seed": 0,
  "identity_manifest_hash": "...",
  "config_hash": "...",
  "python_version": "...",
  "torch_version": "...",
  "transformers_version": "...",
  "peft_version": "...",
  "cuda_version": "...",
  "gpu_name": "...",
  "train_examples": 120,
  "image_exposures": 120,
  "alias_exposures": 0,
  "property_exposures": 120
}

Also:

set Python, NumPy, and PyTorch seeds;

enable deterministic behavior where practical;

record that some CUDA operations may remain nondeterministic;

save train/validation losses;

save candidate-token logits, not just decoded answers;

keep prompts and chat templates fixed;

pin the base model revision.

21. Minimal figures

Produce only figures needed to answer the research question.

Behavioral route preference: conflict marker-following versus identity-following rate.

Ablation dependence: aligned, no-marker, and face-masked accuracy.

Causal route plot: patching restoration by layer for marker, face, text, and final-token groups.

Knockout effects: logit/accuracy change for direct and mediated edge blocks.

Probe curves: identity and property decoding by layer.

Parameter update heat map: layer × module-family update energy.

Adapter ablation: behavioral effect of disabling each layer's adapter.

Avoid dashboards or interactive applications.

22. Execution order

Stage A: Dataset and base checks

Select identities and freeze manifest.

Verify aliases and property strings tokenize acceptably.

Verify base model is near chance on DAX/WUG.

Verify marker bounding-box-to-token mapping.

Verify no image leakage across splits.

Stage B: Establish behavioral separation

Train Direct, one seed.

Train Separated Mediated, one seed.

Verify component competence.

Run aligned/conflict/no-marker/face-masked tests.

Adjust only obvious feasibility issues.

Stage C: Freeze protocol

Freeze training configuration.

Train all three required conditions over three seeds.

Do not change hyperparameters by condition.

Stage D: Mechanistic tests

Run coarse attention knockouts.

Run paired causal patching.

Extract layer activations.

Train probes.

Stage E: Parameter analysis

Merge LoRA deltas virtually.

Compute update distributions.

Run functional adapter-layer ablations.

Stage F: Optional mixed model

Add Mixed only after Direct and Mediated are behaviorally distinguishable.

23. Deliverables

The MVP is complete when it produces:

a fixed identity/property manifest;

deterministic prompt-generation code;

nine required adapters;

one behavior result table;

one pathway-intervention result table;

one probe result table;

one adapter-update result table;

six or seven figures;

a reproducibility manifest;

a short report explaining whether the evidence supports distinct routes.

24. Follow-on unlearning experiment

Do not begin this until the routing MVP succeeds.

After route separation, apply an identical unlearning objective to Direct, Mediated, and Mixed adapters. Then test whether forgetting:

suppresses marker-to-output transfer;

suppresses face-to-identity transfer;

removes alias-to-property knowledge;

causes fallback to the alternate route;

changes the route without deleting the final answer.

The routing MVP provides the necessary pre-unlearning baseline.

25. Primary references and implementation resources

Dataset

CelebA official dataset page: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

CelebA attribute-consistency audit: https://openaccess.thecvf.com/content/CVPR2023W/VDU/html/Wu_Consistency_and_Accuracy_of_CelebA_Attribute_Values_CVPRW_2023_paper.html

Visual information flow

Pathways paper: https://arxiv.org/abs/2607.03358

Pathways source repository: https://github.com/israfelsr/vlm-pathways

Model and training

Qwen3-VL Transformers documentation: https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl

Transformers: https://github.com/huggingface/transformers

PEFT: https://huggingface.co/docs/peft/en/index

LoRA developer guide: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md

Accelerate: https://huggingface.co/docs/accelerate/index

Accelerate gradient accumulation: https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation

bitsandbytes integration: https://huggingface.co/docs/transformers/quantization/bitsandbytes

Hugging Face Datasets image loading: https://huggingface.co/docs/datasets/image_load

Hugging Face Datasets image creation: https://huggingface.co/docs/datasets/image_dataset

Mechanistic analysis

PyTorch module hooks: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html

PyTorch module notes: https://docs.pytorch.org/docs/stable/notes/modules.html

Analysis

scikit-learn linear models: https://scikit-learn.org/stable/modules/linear_model.html

scikit-learn preprocessing: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

scikit-learn PCA: https://scikit-learn.org/stable/modules/decomposition.html

SciPy singular values: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svdvals.html

Safetensors: https://huggingface.co/docs/safetensors/en/index