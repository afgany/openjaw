# Execution-Plan: OpenJaw

> Each step-commit is a single atomic unit of work: code → test → commit.
> Designed so no single step exceeds ~80% of Claude Opus 4.6 context window.
> Every step has: Goal, KPIs, Tests, Files touched, Estimated context usage.

---

## Step 1: Project Scaffold and Config System
**Goal:** Initialize the repo with full directory structure, pyproject.toml, Hydra config, and test infrastructure.

**KPIs:**
- All directories created per PRD-C Section 7.5
- `pyproject.toml` installable with `pip install -e .`
- `pytest` runs and passes (placeholder test)
- Hydra loads `configs/default.yaml` without error

**Tests:**
- `tests/test_scaffold.py`: import openjaw succeeds; config loads

**Files:**
- `pyproject.toml`
- `configs/default.yaml`
- `openjaw/__init__.py` and all sub-package `__init__.py` files
- `.gitignore`
- `README.md` (skeleton)
- `tests/test_scaffold.py`

**Context estimate:** ~20% (boilerplate, small files)

---

## Step 2: MDP Dataclasses and Type System
**Goal:** Implement the formal MDP specification as Python dataclasses — the mathematical contract all modules code against.

**KPIs:**
- All MDP components (S, A, O, R) defined with exact dimensions
- 13-DOF articulator spec with bounds, labels, biomechanical notes
- Type checking passes (mypy or pyright)

**Tests:**
- `tests/test_mdp.py`: state dim = 39, action dim = 13, action bounds = [-0.5, 0.5], observation construction, reward range

**Files:**
- `openjaw/core/mdp.py`
- `openjaw/core/types.py`
- `openjaw/env/articulators.py`
- `tests/test_mdp.py`

**Context estimate:** ~25% (dataclasses + tests)

---

## Step 3: MuJoCo XML — Oral Cavity Geometry
**Goal:** Create the MuJoCo XML model with jaw, tongue (3 segments), lips, teeth, and palate. Tendon-driven actuation for 13 DOF.

**KPIs:**
- MuJoCo loads the XML without error
- 13 actuators defined and controllable
- Simulation steps without divergence (1000 steps, random actions)
- Visually reasonable geometry when rendered

**Tests:**
- `tests/test_env.py::test_mujoco_loads`: model loads
- `tests/test_env.py::test_actuator_count`: 13 actuators
- `tests/test_env.py::test_stability`: 1000 random steps, no NaN

**Files:**
- `openjaw/env/assets/mouth.xml`
- `openjaw/env/oral_cavity.py` (loader + state extraction)
- `tests/test_env.py` (partial)

**Context estimate:** ~40% (XML is verbose, geometry requires care)

---

## Step 4: Gymnasium Environment Wrapper
**Goal:** Wrap the MuJoCo model as a standard Gymnasium environment with correct spaces.

**KPIs:**
- `env.observation_space` matches MDP spec
- `env.action_space` matches MDP spec (Box, 13-dim, bounds)
- `env.reset()` returns valid observation
- `env.step(action)` returns (obs, reward, terminated, truncated, info)
- Vectorized env (16 parallel) works

**Tests:**
- `tests/test_env.py::test_gym_interface`: reset, step, spaces
- `tests/test_env.py::test_vectorized`: 16 parallel envs step correctly

**Files:**
- `openjaw/env/mouth_env.py`
- `openjaw/env/__init__.py` (register env)
- `tests/test_env.py` (extended)

**Context estimate:** ~30%

---

## Step 5: SPARC Audio Decoder Integration
**Goal:** Integrate SPARC to convert 13-DOF articulatory state to audio waveform.

**KPIs:**
- SPARC model loads (or mock if not available — design the interface)
- Articulatory state adapter: MuJoCo 13-DOF → SPARC input format
- Produces 16kHz audio waveform from articulatory trajectory
- Inference < 5ms per step on GPU

**Tests:**
- `tests/test_audio.py::test_sparc_loads`: model loads or mock validates interface
- `tests/test_audio.py::test_output_format`: 16kHz waveform, correct duration
- `tests/test_audio.py::test_adapter`: MuJoCo state converts to SPARC input

**Files:**
- `openjaw/audio/sparc_decoder.py`
- `openjaw/audio/__init__.py`
- `tests/test_audio.py`

**Context estimate:** ~35% (adapter logic + SPARC API investigation)

---

## Step 6: Visual Rendering and Vertex Extraction
**Goal:** Implement MuJoCo offscreen rendering and lip vertex extraction for LVE computation.

**KPIs:**
- Offscreen render produces 256×256 RGB image
- Lip vertices extractable from MuJoCo mesh data
- FLAME-compatible vertex mapping documented (even if approximate)

**Tests:**
- `tests/test_visual.py::test_render_shape`: image is 256×256×3
- `tests/test_visual.py::test_vertex_extraction`: lip vertices are numpy array of correct shape

**Files:**
- `openjaw/visual/renderer.py`
- `openjaw/visual/flame_adapter.py`
- `tests/test_visual.py`

**Context estimate:** ~25%

---

## Step 7: Perception Encoders (Sylber + wav2vec)
**Goal:** Wrap frozen pre-trained audio and visual encoders as perception modules.

**KPIs:**
- Sylber encoder: audio → embedding (correct dim)
- wav2vec encoder: audio → frame features (correct dim)
- Lip reader / visual encoder: vertices → feature vector
- All frozen (no gradient computation)
- Total inference < 10ms per step

**Tests:**
- `tests/test_perception.py::test_sylber`: output shape
- `tests/test_perception.py::test_wav2vec`: output shape
- `tests/test_perception.py::test_visual_encoder`: output shape

**Files:**
- `openjaw/perception/sylber.py`
- `openjaw/perception/wav2vec.py`
- `openjaw/perception/lip_reader.py`
- `tests/test_perception.py`

**Context estimate:** ~35% (model loading + wrapping + mocks)

---

## Step 8: Reward Function Module
**Goal:** Implement the multi-modal reward with ablation toggles.

**KPIs:**
- R_audio: cosine similarity in Sylber space, range [-1, 1]
- R_visual: 1 - normalized LVE, range [0, 1]
- R_combined: weighted sum, configurable via YAML
- R_aux: silence penalty + smoothness + energy
- Ablation: can disable audio or visual independently

**Tests:**
- `tests/test_reward.py::test_audio_reward_range`: output in [-1, 1]
- `tests/test_reward.py::test_visual_reward_range`: output in [0, 1]
- `tests/test_reward.py::test_combined`: weighted correctly
- `tests/test_reward.py::test_ablation_toggle`: audio-only, visual-only modes

**Files:**
- `openjaw/reward/audio_reward.py`
- `openjaw/reward/visual_reward.py`
- `openjaw/reward/auxiliary.py`
- `openjaw/reward/combined.py`
- `tests/test_reward.py`

**Context estimate:** ~30%

---

## Step 9: Policy Network (MLP Actor-Critic)
**Goal:** Implement the actor-critic networks matching the architecture spec.

**KPIs:**
- Actor: obs_dim → 256 → 256 → 13, tanh output
- Critic: obs_dim → 256 → 256 → 1
- Orthogonal init
- Total params < 1M
- Forward pass produces correct output shapes

**Tests:**
- `tests/test_policy.py::test_actor_output`: shape (batch, 13), range [-1, 1]
- `tests/test_policy.py::test_critic_output`: shape (batch, 1)
- `tests/test_policy.py::test_param_count`: < 1M

**Files:**
- `openjaw/policy/networks.py`
- `tests/test_policy.py`

**Context estimate:** ~20%

---

## Step 10: PPO Training Loop
**Goal:** Implement the training loop connecting env → policy → reward → logging.

**KPIs:**
- PPO runs for 10 episodes without error
- Reward logged to TensorBoard
- Checkpoint saves and loads correctly
- Curriculum phase switching works

**Tests:**
- `tests/test_training.py::test_one_episode`: completes without error
- `tests/test_training.py::test_checkpoint`: save/load/resume produces same state
- `tests/test_training.py::test_curriculum`: phase transitions at correct episode counts

**Files:**
- `openjaw/training/trainer.py`
- `openjaw/training/curriculum.py`
- `openjaw/training/logger.py`
- `configs/ppo_syllable.yaml`
- `configs/curriculum.yaml`
- `scripts/train.py`
- `tests/test_training.py`

**Context estimate:** ~45% (integration of all modules)

---

## Step 11: Data Pipeline (VOCASET + Custom)
**Goal:** Implement reference data loading and preprocessing.

**KPIs:**
- VOCASET loader reads audio + FLAME params (or mocks if not downloaded)
- Custom speaker loader reads audio + video
- Preprocessing: segmentation into syllables, embedding extraction
- Target embeddings available for reward computation

**Tests:**
- `tests/test_data.py::test_vocaset_loader`: loads mock data correctly
- `tests/test_data.py::test_custom_loader`: loads mock data correctly
- `tests/test_data.py::test_preprocessing`: segments and extracts embeddings

**Files:**
- `openjaw/data/vocaset.py`
- `openjaw/data/custom_speaker.py`
- `openjaw/data/preprocessing.py`
- `tests/test_data.py`

**Context estimate:** ~30%

---

## Step 12: Evaluation Metrics
**Goal:** Implement all evaluation metrics: MCD, LVE, CER, Sylber similarity.

**KPIs:**
- MCD computes between two audio signals
- LVE computes between two vertex sets
- CER via an ASR model (Whisper or similar)
- Sylber similarity between two audio signals

**Tests:**
- `tests/test_metrics.py`: each metric returns scalar, known input/output pairs validated

**Files:**
- `openjaw/evaluation/metrics.py`
- `scripts/evaluate.py`
- `tests/test_metrics.py`

**Context estimate:** ~25%

---

## Step 13: Ablation Framework
**Goal:** Implement experiment runner for ablation studies.

**KPIs:**
- Run training with audio-only reward config
- Run training with visual-only reward config
- Run training with no-curriculum config
- Results collected in structured format

**Tests:**
- `tests/test_ablations.py`: ablation configs load, reward components toggle correctly

**Files:**
- `openjaw/evaluation/ablations.py`
- `configs/ablations/audio_only.yaml`
- `configs/ablations/visual_only.yaml`
- `configs/ablations/no_curriculum.yaml`
- `tests/test_ablations.py`

**Context estimate:** ~25%

---

## Step 14: Visualization Tools
**Goal:** Implement plotting for reward curves, articulator trajectories, spectrograms, and side-by-side comparisons.

**KPIs:**
- Reward curve plot from TensorBoard logs
- Articulator trajectory plot (13 DOF over time)
- Audio spectrogram comparison (generated vs. target)
- All plots saved as PDF for paper inclusion

**Tests:**
- Visual inspection (no automated tests — generate sample plots from mock data)

**Files:**
- `openjaw/evaluation/visualization.py`
- `scripts/visualize.py`

**Context estimate:** ~25%

---

## Step 15: Integration Test — Full Pipeline
**Goal:** Run the complete pipeline end-to-end: load data → create env → train 100 episodes → evaluate → visualize.

**KPIs:**
- Full pipeline runs without error on consumer GPU
- Training produces improving reward curve (even slightly)
- Evaluation metrics computed
- Figures generated

**Tests:**
- `tests/test_integration.py`: end-to-end pipeline test (100 episodes, mock data)

**Files:**
- `tests/test_integration.py`
- Bug fixes across all modules

**Context estimate:** ~50% (touches many files for debugging)

---

## Step 16: Training Runs — Babbling + Vowels
**Goal:** Execute curriculum Phase 0 (babbling) and Phase 1 (vowels) on consumer GPU.

**KPIs:**
- Babbling: agent learns to produce some sound (reward > -0.5 average)
- Vowels: 5 cardinal vowels, each reaching Sylber cos_sim > 0.5
- Total GPU time < 20 hours

**Tests:**
- Reward curves show improvement
- Manual listening test: some vowels sound vaguely correct

**Files:**
- Trained checkpoints (saved to `checkpoints/`)
- Training logs

**Context estimate:** ~15% (mostly running scripts, reviewing logs)

---

## Step 17: Training Runs — Syllables
**Goal:** Execute curriculum Phase 2 (syllable imitation) on consumer GPU.

**KPIs:**
- 10-20 syllables trained
- Sylber cos_sim > 0.7 on best syllables
- Human transcription > 50% correct on 10+ syllables
- Total GPU time < 60 hours

**Tests:**
- Automated metrics
- Human evaluation (informal, 3+ raters)

**Files:**
- Trained checkpoints
- Training logs
- Evaluation results

**Context estimate:** ~15%

---

## Step 18: Ablation Experiments
**Goal:** Run ablation studies: audio-only, visual-only, combined, no-curriculum.

**KPIs:**
- All 4 conditions trained for same number of episodes
- Results show combined > audio-only > visual-only (expected)
- Results show curriculum > no-curriculum (expected)
- Tables generated

**Tests:**
- Statistical comparison between conditions

**Files:**
- Ablation checkpoints and logs
- Result tables

**Context estimate:** ~15%

---

## Step 19: Paper Writing — Methods and Architecture
**Goal:** Write the core paper sections: Introduction, Related Work, Methods (including all 7 design decisions).

**KPIs:**
- Introduction frames the problem and contribution clearly
- Related Work covers all 20 references
- Methods section presents MDP, architecture, reward, curriculum with full mathematical notation
- Methods Analysis subsection justifies all 7 design decisions
- Architecture diagram finalized

**Tests:**
- Self-review: all claims supported, notation consistent, no gaps

**Files:**
- `paper/main.tex` (sections 1-4)
- `paper/references.bib`
- `paper/figures/architecture.pdf`

**Context estimate:** ~60% (significant text generation)

---

## Step 20: Paper Writing — Experiments and Discussion
**Goal:** Write Experiments, Results, Discussion, and Conclusion sections.

**KPIs:**
- Experiments section describes setup, baselines, metrics
- Results section presents all tables and figures
- Ablation analysis is clear
- Discussion covers limitations and future work
- Abstract is compelling

**Tests:**
- Self-review: results match experiments, discussion is balanced

**Files:**
- `paper/main.tex` (sections 5-8, abstract)
- `paper/figures/` (result plots)
- `paper/scripts/generate_figures.py`

**Context estimate:** ~60%

---

## Step 21: Paper Figures and Tables
**Goal:** Generate all paper-ready figures and tables from experiment data.

**KPIs:**
- Architecture diagram (vector PDF)
- Reward curves (training progression)
- Articulator trajectory plots
- Spectrogram comparisons
- Ablation comparison table
- Metric summary table

**Tests:**
- Visual inspection: figures are publication quality

**Files:**
- `paper/figures/*.pdf`
- `paper/scripts/generate_figures.py`

**Context estimate:** ~30%

---

## Step 22: Documentation and Release Prep
**Goal:** Complete README, add LICENSE, ensure reproducibility.

**KPIs:**
- README has complete setup/train/evaluate instructions
- LICENSE file present (MIT recommended)
- All configs documented
- Reproducibility: seeds, hardware, expected runtime documented
- All tests pass

**Tests:**
- Fresh clone + install + run tests passes

**Files:**
- `README.md` (complete)
- `LICENSE`
- Final test run

**Context estimate:** ~25%

---

## Summary

| Step | Name | Est. Context | Dependencies |
|------|------|:------------:|-------------|
| 1 | Project Scaffold | 20% | — |
| 2 | MDP Dataclasses | 25% | Step 1 |
| 3 | MuJoCo XML Model | 40% | Step 2 |
| 4 | Gymnasium Env | 30% | Step 3 |
| 5 | SPARC Audio | 35% | Step 4 |
| 6 | Visual Rendering | 25% | Step 4 |
| 7 | Perception Encoders | 35% | Steps 5, 6 |
| 8 | Reward Module | 30% | Step 7 |
| 9 | Policy Network | 20% | Step 2 |
| 10 | PPO Training Loop | 45% | Steps 8, 9 |
| 11 | Data Pipeline | 30% | Step 7 |
| 12 | Evaluation Metrics | 25% | Step 7 |
| 13 | Ablation Framework | 25% | Step 10 |
| 14 | Visualization | 25% | Step 12 |
| 15 | Integration Test | 50% | Steps 10-14 |
| 16 | Training: Babbling+Vowels | 15% | Step 15 |
| 17 | Training: Syllables | 15% | Step 16 |
| 18 | Ablation Experiments | 15% | Step 17 |
| 19 | Paper: Methods | 60% | Step 15 |
| 20 | Paper: Results | 60% | Steps 18, 19 |
| 21 | Paper: Figures | 30% | Step 20 |
| 22 | Documentation | 25% | Step 21 |
