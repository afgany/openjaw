# Dev-Plan: OpenJaw

> Development stages for building the RL visual-acoustic articulatory imitation framework.
> Each stage is a logical unit of work with clear deliverables and dependencies.

---

## Stage 0: Project Scaffolding
**Goal:** Set up the repository, dependency management, config system, and testing infrastructure.
**Deliverables:**
- pyproject.toml with all dependencies
- Directory structure matching PRD-C Section 7.5
- Hydra config system with default.yaml
- pytest infrastructure with a passing placeholder test
- Git init with .gitignore
- README.md skeleton

**Dependencies:** None
**Risk:** Low

---

## Stage 1: MDP Formulation Module
**Goal:** Implement the mathematical MDP as Python dataclasses — the formal specification that all other modules implement against.
**Deliverables:**
- `openjaw/core/mdp.py` — StateSpace, ActionSpace, ObservationSpace, RewardSpec, TransitionSpec dataclasses
- `openjaw/core/types.py` — shared type aliases and constants (DOF count, bounds, dimensions)
- `openjaw/env/articulators.py` — 13-DOF articulator definition with bounds and labels
- Unit tests: `tests/test_mdp.py` — validates dimensions, bounds, serialization

**Dependencies:** Stage 0
**Risk:** Low — pure specification code

---

## Stage 2: MuJoCo Oral Cavity Model
**Goal:** Build the MuJoCo XML model of the oral cavity with 13-DOF tendon-driven actuation and wrap it as a Gymnasium environment.
**Deliverables:**
- `openjaw/env/assets/mouth.xml` — MuJoCo XML: jaw (rigid), tongue (3 segments, tendon-actuated), lips (upper/lower, tendon-actuated), teeth (static), palate (static)
- `openjaw/env/oral_cavity.py` — model loader, state extraction, actuation mapping
- `openjaw/env/mouth_env.py` — Gymnasium Env: reset(), step(), render(), observation/action spaces
- Unit tests: `tests/test_env.py` — env creation, step, reset, action bounds, observation shape
- Visualization: render a few frames to verify geometry

**Dependencies:** Stage 1 (uses MDP specs for spaces)
**Risk:** Medium — MuJoCo XML authoring requires iteration for anatomically reasonable geometry

---

## Stage 3: SPARC Audio Integration
**Goal:** Integrate the SPARC decoder to convert articulatory trajectories to audio waveforms.
**Deliverables:**
- `openjaw/audio/sparc_decoder.py` — load pre-trained SPARC, adapt MuJoCo state → SPARC input format, inference
- `openjaw/audio/glottal.py` — glottal source model (if SPARC requires separate voicing input)
- Unit tests: `tests/test_audio.py` — SPARC loads, produces audio from known trajectory, output shape/rate correct
- Validation: play back audio from a manually set articulator trajectory

**Dependencies:** Stage 2 (needs articulatory state format)
**Risk:** Medium — SPARC input format may need an adapter from MuJoCo state representation

---

## Stage 4: Visual Rendering Pipeline
**Goal:** Extract visual output from MuJoCo simulation for reward computation.
**Deliverables:**
- `openjaw/visual/renderer.py` — MuJoCo offscreen rendering, image capture
- `openjaw/visual/flame_adapter.py` — extract lip vertices from MuJoCo mesh, map to FLAME-compatible format for LVE
- Unit tests: `tests/test_visual.py` — rendering produces images of correct size, vertex extraction works

**Dependencies:** Stage 2 (needs MuJoCo model)
**Risk:** Low-Medium — MuJoCo rendering is well-documented; FLAME mapping needs careful vertex correspondence

---

## Stage 5: Perception Encoders
**Goal:** Wrap frozen pre-trained models (Sylber, wav2vec 2.0) as perception modules.
**Deliverables:**
- `openjaw/perception/sylber.py` — Sylber encoder wrapper (audio → syllabic embedding)
- `openjaw/perception/wav2vec.py` — wav2vec 2.0 wrapper (audio → frame features)
- `openjaw/perception/lip_reader.py` — visual feature extractor (mesh → lip features)
- Unit tests: `tests/test_perception.py` — each encoder produces correct output dimensions from dummy input

**Dependencies:** Stage 3 (needs audio output format), Stage 4 (needs visual output format)
**Risk:** Medium — Sylber availability and API may need investigation

---

## Stage 6: Reward Module
**Goal:** Implement the multi-modal reward function with ablation support.
**Deliverables:**
- `openjaw/reward/audio_reward.py` — Sylber cosine similarity
- `openjaw/reward/visual_reward.py` — normalized LVE
- `openjaw/reward/auxiliary.py` — silence penalty, smoothness, energy
- `openjaw/reward/combined.py` — weighted combination with config-driven weights, ablation toggles
- Unit tests: `tests/test_reward.py` — reward ranges, ablation toggles, gradient of reward w.r.t. inputs

**Dependencies:** Stage 5 (needs encoders)
**Risk:** Low — composition of existing components

---

## Stage 7: Policy Network and PPO Training
**Goal:** Implement the actor-critic policy network and PPO training loop.
**Deliverables:**
- `openjaw/policy/networks.py` — MLP Actor (obs→256→256→13, tanh), MLP Critic (obs→256→256→1)
- `openjaw/training/trainer.py` — PPO training loop using SB3 or CleanRL
- `openjaw/training/curriculum.py` — phase scheduler (babbling → vowels → syllables)
- `openjaw/training/logger.py` — TensorBoard logging wrapper
- `configs/ppo_syllable.yaml`, `configs/curriculum.yaml`
- Unit tests: `tests/test_training.py` — one episode runs, loss computes, checkpoint saves/loads

**Dependencies:** Stage 6 (needs reward), Stage 2 (needs env)
**Risk:** Medium — integration of all modules; debugging reward signals

---

## Stage 8: Data Pipeline
**Goal:** Load and preprocess reference speaker data for imitation targets.
**Deliverables:**
- `openjaw/data/vocaset.py` — VOCASET loader (audio + FLAME params)
- `openjaw/data/custom_speaker.py` — custom recording loader (audio + video → embeddings)
- `openjaw/data/preprocessing.py` — segment into syllables, extract embeddings
- Unit tests: basic loader tests with mock data

**Dependencies:** Stage 5 (needs encoders for embedding extraction)
**Risk:** Low-Medium — VOCASET download and format

---

## Stage 9: Evaluation Suite
**Goal:** Implement all evaluation metrics and visualization tools.
**Deliverables:**
- `openjaw/evaluation/metrics.py` — MCD, LVE, CER (via ASR), Sylber similarity
- `openjaw/evaluation/ablations.py` — ablation experiment runner
- `openjaw/evaluation/visualization.py` — reward curves, articulator trajectories, spectrograms
- `scripts/evaluate.py` — evaluation entry point
- `scripts/visualize.py` — visualization entry point

**Dependencies:** Stage 7 (needs trained checkpoints), Stage 8 (needs test data)
**Risk:** Low

---

## Stage 10: Training Runs and Experiments
**Goal:** Execute the curriculum training, ablations, and collect all results.
**Deliverables:**
- Trained checkpoints for each curriculum phase
- Ablation results: audio-only, visual-only, combined, no-curriculum
- Evaluation metrics on VOCASET test set
- Generated figures and tables

**Dependencies:** All previous stages
**Risk:** Medium — training may need hyperparameter tuning within compute budget

---

## Stage 11: Scientific Paper
**Goal:** Write the complete paper with all results.
**Deliverables:**
- `paper/main.tex` — complete paper (NeurIPS/ICML format)
- `paper/references.bib` — bibliography
- `paper/figures/` — all figures (architecture, results, ablations)
- `paper/scripts/generate_figures.py` — automated figure generation

**Dependencies:** Stage 10 (needs results)
**Risk:** Low (writing), Medium (framing and positioning)

---

## Stage 12: Documentation and Release
**Goal:** Prepare the repository for public release.
**Deliverables:**
- README.md — complete setup, training, evaluation instructions
- LICENSE file
- Reproducibility documentation (seeds, configs, hardware)
- CI: linting and test workflow

**Dependencies:** Stage 11
**Risk:** Low

---

## Dependency Graph

```
Stage 0 (Scaffold)
    │
    ▼
Stage 1 (MDP)
    │
    ▼
Stage 2 (MuJoCo Env)
    │
    ├──────────────┬──────────────┐
    ▼              ▼              ▼
Stage 3         Stage 4       (parallel)
(SPARC Audio)   (Visual)
    │              │
    └──────┬───────┘
           ▼
    Stage 5 (Perception)
           │
           ▼
    Stage 6 (Reward)
           │
           ▼
    Stage 7 (Policy + PPO)    Stage 8 (Data)
           │                      │
           └──────────┬───────────┘
                      ▼
              Stage 9 (Evaluation)
                      │
                      ▼
             Stage 10 (Training Runs)
                      │
                      ▼
             Stage 11 (Paper)
                      │
                      ▼
             Stage 12 (Release)
```
