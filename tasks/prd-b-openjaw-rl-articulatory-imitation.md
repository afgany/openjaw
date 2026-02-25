# PRD-B: OpenJaw — Reinforcement Learning for Embodied Speech Imitation in Multi-Physics 3D Simulation

> **Version:** PRD-B (Learning Phase Analysis Integrated)
> **Date:** 2026-02-24
> **Status:** Awaiting clarification questions before PRD-C

---

## 1. Introduction/Overview

OpenJaw is a research project that develops a **mathematical RL framework for closed-loop embodied speech imitation**, trained entirely within a multi-physics 3D simulation with audio-visual sensory feedback. The deliverables are a **GitHub repository** (code, models, configs, evaluation) and a **scientific paper**.

### The Core Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP ARCHITECTURE                  │
│                                                              │
│  Reference       ┌──────────┐    Control     ┌───────────┐  │
│  Audio/Video ──▶ │ RL Policy │ ──signals──▶  │ 3D Multi- │  │
│                  │ (Actor-  │               │ Physics    │  │
│                  │  Critic) │               │ Mouth Sim  │  │
│                  └────▲─────┘               └─────┬──────┘  │
│                       │                           │          │
│              Reward   │    ┌──────────────┐       │ Physics  │
│              Signal   │    │ Reward       │       │ Output   │
│                       ├────│ Computation  │◀──┐   │          │
│                       │    └──────────────┘   │   ▼          │
│                       │                  ┌────┴──────────┐   │
│                       │                  │ Virtual       │   │
│                       └──────────────────│ Sensors       │   │
│                         Observations     │ (Camera+Mic)  │   │
│                                          └───────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Problem Statement

Current approaches to speech synthesis bypass the physical process of speech production. Text-to-speech (TTS) and audio-driven talking-head systems map signals directly to waveforms or mesh deformations without modeling the underlying biomechanics. This creates fundamental limitations:

1. **No physical grounding:** Generated speech doesn't emerge from physical articulation, limiting scientific understanding of speech motor control
2. **No closed-loop learning:** Existing systems are open-loop (feedforward), unlike human speech acquisition which relies on auditory and somatosensory feedback (DIVA model, Guenther Lab)
3. **No embodiment:** Cannot transfer to physical systems (robots, prosthetics) because the generation process is non-physical
4. **No person-specific imitation:** Current articulatory RL (Anand et al. 2025) achieves syllable-level control but not person-specific style imitation

OpenJaw closes these gaps by building the **first system that combines RL-driven articulatory control with full multi-physics 3D simulation producing both audio AND visual output in a closed feedback loop**.

### Novelty and Scientific Contribution

Based on extensive literature analysis, OpenJaw's novelty lies at the intersection of three areas where no prior work exists:

| Existing Work | What They Do | What's Missing |
|---------------|-------------|----------------|
| Anand et al. 2025 | RL + articulatory control → audio | No physics sim, no visual output, no person-specific imitation |
| Shitov et al. 2023 | Model-based RL + VocalTractLab → vowels | 2D geometric model, no visual, no multi-physics |
| Nguyen et al. 2022 | RL + FEM facial sim | No audio output, facial symmetry only, no speech |
| Hu et al. 2026 (Columbia) | Self-supervised lip sync on robot | Physical robot (not sim), no RL, audio-to-motion only |
| SelfTalk 2023 | Audio → 3D face animation | No physics, no RL, mesh deformation only |
| Disney/ETH 2024 | Physical face model (FEM) | No RL, no audio, identity fitting only |

**OpenJaw uniquely combines:** RL policy optimization + multi-physics articulatory simulation + acoustic wave generation + visual rendering + closed-loop sensory feedback + person-specific imitation.

---

## 2. Goals

- **G1:** Formalize the mathematical RL framework (MDP formulation, state/action/observation/reward spaces) for articulatory speech imitation
- **G2:** Implement a multi-physics 3D simulation of the human oral cavity producing both acoustic and visual output
- **G3:** Demonstrate closed-loop RL training where the agent improves speech imitation through audio-visual feedback
- **G4:** Achieve intelligible speech production (>80% correct human transcription on test syllables)
- **G5:** Demonstrate person-specific imitation (measurable audio-visual similarity to target speaker)
- **G6:** Publish at a top-tier venue (NeurIPS, ICML, ICLR, ICRA, CoRL, or Interspeech)
- **G7:** Release reproducible open-source codebase

---

## 3. User Stories

### US-001: MDP Formulation and Mathematical Framework
**Description:** As a researcher, I want a rigorous MDP formulation for the articulatory speech imitation problem, so that the RL framework has solid mathematical foundations for the paper.

**Acceptance Criteria:**
- [ ] State space S defined: articulator positions + velocities + acoustic features + visual features
- [ ] Action space A defined: continuous muscle activation signals (bounded)
- [ ] Observation space O defined: encoded audio + visual from virtual sensors
- [ ] Reward function R defined: multi-modal similarity to target + auxiliary shaping terms
- [ ] Transition dynamics T defined: physics simulation step
- [ ] Discount factor, episode structure, and horizon documented
- [ ] Mathematical notation consistent throughout codebase and paper

### US-002: Articulatory Model Definition
**Description:** As a researcher, I want a biomechanically motivated articulatory model defining all controllable degrees of freedom, so that the RL agent has a well-specified action space.

**Acceptance Criteria:**
- [ ] Tongue model: minimum 6 DOF (dorsum, blade, tip in X/Y; inspired by Anand et al.)
- [ ] Jaw model: 3 DOF (open/close, protrusion, lateral)
- [ ] Lip model: 4-6 DOF (upper/lower protrusion, rounding, spread)
- [ ] Velum: 1 DOF (open/close for nasality)
- [ ] Larynx: 2 DOF (pitch, voicing amplitude)
- [ ] Total: 16-33 continuous control parameters documented with biomechanical justification
- [ ] Muscle activation dynamics (Hill-type or simplified) with activation/deactivation time constants

### US-003: Physics Simulation Environment
**Description:** As a researcher, I want a GPU-accelerated physics simulation of the oral cavity that runs as a Gymnasium-compatible environment, so that standard RL libraries can train on it.

**Acceptance Criteria:**
- [ ] 3D mesh of oral cavity (jaw, tongue, lips, teeth, palate, pharynx)
- [ ] Soft-body physics for tongue (FEM or PBD)
- [ ] Soft-body physics for lips (FEM or PBD)
- [ ] Rigid-body dynamics for jaw, teeth, skull
- [ ] Muscle-tendon actuation converting control signals to forces
- [ ] Contact dynamics (tongue-palate, tongue-teeth, lip-lip)
- [ ] Gymnasium `Env` interface: `reset()`, `step(action)`, `render()`
- [ ] Parallel environment vectorization (minimum 16 simultaneous environments)
- [ ] Simulation timestep: 1-2ms physics, 40ms control (25 Hz control frequency)
- [ ] GPU acceleration via Isaac Lab, MuJoCo MJX, or NVIDIA Warp

### US-004: Acoustic Output Module
**Description:** As a researcher, I want the simulation to produce acoustic output from the vocal tract geometry at each control step, so that the RL agent receives audio feedback.

**Acceptance Criteria:**
- [ ] Acoustic output generated from vocal tract configuration (geometry → sound)
- [ ] Method selection: SPARC neural decoder OR physics-based acoustic propagation
- [ ] Output: waveform at 16kHz minimum sample rate
- [ ] Latency: acoustic output available within same control step
- [ ] Supports both voiced and unvoiced sounds
- [ ] Glottal source model (pitch-synchronous pulse or LF model)

### US-005: Visual Rendering Module
**Description:** As a researcher, I want the simulation to render visual frames of the mouth at each control step, so that the RL agent receives visual feedback.

**Acceptance Criteria:**
- [ ] 3D rendering of mouth region at each timestep
- [ ] FLAME-compatible mesh topology (5023 vertices) for evaluation compatibility
- [ ] Minimum 256x256 resolution
- [ ] 25 FPS minimum (matching control frequency)
- [ ] Supports both mesh rendering and photorealistic rendering (3D Gaussian Splatting optional)
- [ ] Camera position configurable (frontal, profile)

### US-006: Virtual Sensor System
**Description:** As a researcher, I want virtual cameras and microphones that encode the simulation output into feature vectors, so that these can be used for observation and reward computation.

**Acceptance Criteria:**
- [ ] Virtual microphone: captures acoustic output, encodes via wav2vec 2.0 or Sylber
- [ ] Virtual camera: captures visual frames, encodes via visual encoder (e.g., AV-HuBERT, lip-reading net)
- [ ] Observation vector: concatenation of encoded audio + visual features
- [ ] Feature dimensions documented and configurable
- [ ] Encoding runs on GPU with batch support

### US-007: Reward Function Implementation
**Description:** As a researcher, I want a multi-modal reward function that measures imitation quality, so that the RL agent optimizes toward faithful person-specific speech reproduction.

**Acceptance Criteria:**
- [ ] Audio reward: cosine similarity in Sylber embedding space (following Anand et al.)
- [ ] Visual reward: lip vertex error (LVE) against target facial landmarks/mesh
- [ ] Combined reward: R = w_a * R_audio + w_v * R_visual + R_auxiliary
- [ ] Auxiliary rewards: silence penalty, smoothness bonus, energy regularization
- [ ] Configurable weights (w_a, w_v) via config file
- [ ] Commutative verification reward (SelfTalk paradigm): lip-reading accuracy on generated visual

### US-008: RL Training Pipeline
**Description:** As a researcher, I want an end-to-end training pipeline with experiment management, so that I can run reproducible experiments efficiently.

**Acceptance Criteria:**
- [ ] PPO implementation (primary) via Stable-Baselines3 or CleanRL
- [ ] SAC implementation (secondary) for comparison
- [ ] Model-based RL variant (optional, for sample efficiency comparison)
- [ ] Hyperparameter configuration via YAML files
- [ ] Curriculum learning support: babbling → vowels → syllables → words
- [ ] Training logging to TensorBoard and/or Weights & Biases
- [ ] Checkpointing every N episodes with resumable training
- [ ] Seed control for reproducibility
- [ ] Multi-GPU support for parallel environments

### US-009: Reference Data Pipeline
**Description:** As a researcher, I want to load and process reference audio-video data of a target speaker, so that the RL agent has imitation targets.

**Acceptance Criteria:**
- [ ] Load audio files (WAV, 16kHz+)
- [ ] Load video files (MP4) or 3D facial motion capture (FLAME parameters)
- [ ] Extract per-frame audio embeddings (Sylber/wav2vec 2.0)
- [ ] Extract per-frame visual features (facial landmarks, FLAME parameters, lip region)
- [ ] Segment into episodes (syllable-level and/or utterance-level)
- [ ] Support VOCASET and BIWI datasets for benchmarking

### US-010: Evaluation Suite
**Description:** As a researcher, I want comprehensive evaluation metrics and visualization tools, so that I can quantify results and generate paper figures.

**Acceptance Criteria:**
- [ ] Mel-Cepstral Distortion (MCD) between generated and target audio
- [ ] Lip Vertex Error (LVE) on FLAME mesh
- [ ] Upper-Face Dynamics Deviation (FDD)
- [ ] Character Error Rate (CER) via ASR on generated audio
- [ ] Human evaluation protocol: Mean Opinion Score (MOS) for naturalness
- [ ] Perceptual similarity (Sylber cosine similarity)
- [ ] Articulator trajectory visualization (2D plots over time)
- [ ] Side-by-side video comparison (generated vs. target)
- [ ] Ablation: audio-only reward vs. visual-only vs. combined

### US-011: Scientific Paper
**Description:** As a researcher, I want the codebase to support reproducible experiments and generate paper-ready artifacts, so that results can be submitted for peer review.

**Acceptance Criteria:**
- [ ] LaTeX paper template (NeurIPS/ICML format)
- [ ] Experiment config files for every result in the paper
- [ ] Figure generation scripts (matplotlib/seaborn)
- [ ] Table generation scripts
- [ ] Reproducibility checklist completed
- [ ] README with setup, training, and evaluation instructions

---

## 4. Functional Requirements

### RL Core
- **FR-1:** Implement PPO with MLP actor-critic networks for continuous articulatory control
- **FR-2:** Implement SAC with twin Q-networks as comparison algorithm
- **FR-3:** State representation: frame-stacked articulator positions + velocities (last 15 frames, ~195-390 dimensions)
- **FR-4:** Action space: 16-33 dimensional continuous vector (muscle activations), bounded [-1, 1]
- **FR-5:** Episode structure: configurable length (default 50 steps = ~2 seconds at 25Hz)
- **FR-6:** Curriculum: progressive difficulty from babbling → single vowels → CV syllables → CVC words → continuous speech

### Simulation
- **FR-7:** 3D oral cavity mesh with anatomically motivated geometry
- **FR-8:** Soft-body tongue simulation with minimum 6 independent muscle groups
- **FR-9:** Deformable lip simulation with independent upper/lower lip control
- **FR-10:** Rigid jaw with 3-DOF articulation
- **FR-11:** Contact handling for tongue-palate, tongue-teeth, lip-lip interactions
- **FR-12:** Simulation timestep ≤ 2ms for physics stability; control timestep = 40ms
- **FR-13:** Vectorized parallel environments (minimum 16, target 64+)

### Audio
- **FR-14:** Generate audio waveform from vocal tract configuration each control step
- **FR-15:** Support both neural decoder (SPARC) and parametric synthesizer (VTL) approaches
- **FR-16:** Glottal source with controllable F0 and voicing amplitude
- **FR-17:** Output sample rate: minimum 16kHz

### Visual
- **FR-18:** Render 3D mouth mesh each control step
- **FR-19:** FLAME-compatible mesh for evaluation against baselines
- **FR-20:** Configurable camera (position, FOV, resolution)

### Perception & Reward
- **FR-21:** Sylber encoder for syllabic audio embeddings
- **FR-22:** wav2vec 2.0 encoder for frame-level audio features
- **FR-23:** Lip-reading network for visual verification (SelfTalk commutative paradigm)
- **FR-24:** Multi-modal reward: R = w_a * cos_sim(sylber_gen, sylber_target) + w_v * (1 - LVE_normalized)
- **FR-25:** Silence penalty: -1 if no syllable detected

### Training Infrastructure
- **FR-26:** Config-driven experiments (YAML/JSON)
- **FR-27:** TensorBoard + W&B logging
- **FR-28:** Model checkpointing with best-model tracking
- **FR-29:** Multi-GPU training support
- **FR-30:** Reproducible seeding

### Evaluation
- **FR-31:** MCD, LVE, FDD, CER metrics
- **FR-32:** MOS human evaluation protocol
- **FR-33:** Baseline comparisons (direct neural TTS, VTL, SPARC-only)
- **FR-34:** Ablation study framework

---

## 5. Non-Goals (Out of Scope)

- Physical robot construction or control
- Real-time production TTS system
- Text input (input is reference audio/video only)
- Emotional expression beyond speech articulation
- Full-body gesture generation
- Language-specific phoneme engineering
- Cloud deployment or web interface
- Sim-to-real transfer (future work, but architecture should not preclude it)
- Training on more than one target speaker simultaneously (single-speaker imitation first)
- End-to-end differentiable physics (use RL, not backprop through simulation)

---

## 6. Architecture Design

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OPENJAW SYSTEM ARCHITECTURE                  │
│                                                                      │
│  ┌─────────────┐     ┌─────────────────────────────────────────┐    │
│  │ Reference    │     │           RL TRAINING LOOP               │    │
│  │ Data        │     │                                          │    │
│  │ Pipeline    │     │  ┌──────────┐      ┌───────────────┐    │    │
│  │             │     │  │ Policy   │      │ Simulation    │    │    │
│  │ Audio ──────┤────▶│  │ Network  │─act─▶│ Environment   │    │    │
│  │ Video ──────┤     │  │ (PPO/SAC)│      │               │    │    │
│  │ FLAME ──────┤     │  │          │◀─obs─│ ┌───────────┐ │    │    │
│  │             │     │  └──────────┘      │ │ Physics   │ │    │    │
│  └─────────────┘     │       ▲            │ │ Engine    │ │    │    │
│                      │       │ reward     │ └─────┬─────┘ │    │    │
│  ┌─────────────┐     │  ┌────┴─────┐     │       │       │    │    │
│  │ Evaluation  │     │  │ Reward   │     │  ┌────▼────┐  │    │    │
│  │ Suite       │     │  │ Module   │◀────│  │ Audio   │  │    │    │
│  │             │     │  │          │     │  │ Module  │  │    │    │
│  │ MCD, LVE,  │     │  └──────────┘     │  └─────────┘  │    │    │
│  │ CER, MOS   │     │       ▲            │  ┌─────────┐  │    │    │
│  │             │     │       │            │  │ Visual  │  │    │    │
│  └─────────────┘     │  ┌────┴─────┐     │  │ Module  │  │    │    │
│                      │  │ Sensor   │◀────│  └─────────┘  │    │    │
│  ┌─────────────┐     │  │ Encoders │     │               │    │    │
│  │ Paper       │     │  │(Sylber,  │     └───────────────┘    │    │
│  │ Artifacts   │     │  │ wav2vec) │                          │    │
│  └─────────────┘     │  └──────────┘                          │    │
│                      └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Repository Structure

```
OpenJaw/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml              # Default training config
│   ├── ppo_syllable.yaml         # PPO syllable-level training
│   ├── sac_syllable.yaml         # SAC comparison
│   ├── curriculum.yaml           # Curriculum schedule
│   └── eval.yaml                 # Evaluation config
├── openjaw/
│   ├── __init__.py
│   ├── envs/                     # Gymnasium environments
│   │   ├── __init__.py
│   │   ├── mouth_env.py          # Main Gymnasium env wrapper
│   │   ├── physics/              # Physics simulation
│   │   │   ├── __init__.py
│   │   │   ├── oral_cavity.py    # Mesh + physics setup
│   │   │   ├── tongue.py         # Tongue soft-body model
│   │   │   ├── lips.py           # Lip deformable model
│   │   │   ├── jaw.py            # Jaw rigid-body model
│   │   │   ├── muscles.py        # Muscle actuation model
│   │   │   └── contact.py        # Contact handling
│   │   ├── audio/                # Acoustic output
│   │   │   ├── __init__.py
│   │   │   ├── synthesizer.py    # Audio from vocal tract
│   │   │   ├── glottal.py        # Glottal source model
│   │   │   └── sparc_decoder.py  # SPARC neural decoder
│   │   └── visual/               # Visual rendering
│   │       ├── __init__.py
│   │       ├── renderer.py       # 3D mesh rendering
│   │       └── flame_adapter.py  # FLAME mesh compatibility
│   ├── agents/                   # RL algorithms
│   │   ├── __init__.py
│   │   ├── ppo.py                # PPO agent
│   │   ├── sac.py                # SAC agent
│   │   └── networks.py           # Actor-critic networks
│   ├── rewards/                  # Reward computation
│   │   ├── __init__.py
│   │   ├── audio_reward.py       # Sylber-based audio reward
│   │   ├── visual_reward.py      # LVE-based visual reward
│   │   ├── combined_reward.py    # Multi-modal reward
│   │   └── auxiliary.py          # Penalties and shaping
│   ├── perception/               # Sensor encoders
│   │   ├── __init__.py
│   │   ├── sylber_encoder.py     # Syllabic embeddings
│   │   ├── wav2vec_encoder.py    # Speech features
│   │   └── lip_reader.py         # Visual verification
│   ├── data/                     # Reference data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py            # Audio-video dataset
│   │   ├── vocaset.py            # VOCASET loader
│   │   └── preprocessing.py      # Feature extraction
│   ├── training/                 # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── curriculum.py         # Curriculum scheduler
│   │   └── logger.py             # TensorBoard/W&B
│   └── evaluation/               # Evaluation suite
│       ├── __init__.py
│       ├── metrics.py            # MCD, LVE, FDD, CER
│       ├── human_eval.py         # MOS protocol
│       ├── visualization.py      # Trajectory plots
│       └── comparison.py         # Baseline comparison
├── paper/
│   ├── main.tex                  # LaTeX paper
│   ├── figures/                  # Generated figures
│   └── tables/                   # Generated tables
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── visualize.py              # Visualization
│   └── generate_figures.py       # Paper figures
├── tests/
│   ├── test_env.py
│   ├── test_physics.py
│   ├── test_audio.py
│   ├── test_reward.py
│   └── test_training.py
└── tasks/                        # PRD and planning docs
    ├── prd-a-openjaw-rl-articulatory-imitation.md
    └── prd-b-openjaw-rl-articulatory-imitation.md
```

### 6.3 MDP Formulation

**State Space S:**
```
s_t = [q_t, q̇_t, a_{t-1}] ∈ ℝ^d_s

where:
  q_t    = articulator positions (16-33 DOF)
  q̇_t   = articulator velocities (16-33 DOF)
  a_{t-1} = previous action (16-33 DOF)
  d_s    = 3 × DOF (48-99 dimensions)
```

**Observation Space O:**
```
o_t = [s_t^{(t-K:t)}, e_audio_t, e_visual_t, e_target_t] ∈ ℝ^d_o

where:
  s_t^{(t-K:t)} = frame-stacked state (K=15 frames)
  e_audio_t      = Sylber/wav2vec embedding of generated audio
  e_visual_t     = visual feature of generated mouth image
  e_target_t     = target audio-visual embedding for current segment
  d_o            ≈ K × d_s + d_audio + d_visual + d_target
```

**Action Space A:**
```
a_t ∈ [-1, 1]^d_a

where d_a = number of muscle activation channels (16-33)
Actions are muscle activations, scaled to force by the muscle model.
```

**Transition Dynamics T:**
```
s_{t+1} = PhysicsStep(s_t, a_t, Δt)

Δt_physics = 1-2ms (simulation substeps)
Δt_control = 40ms (RL decision frequency = 25 Hz)
Substeps per control step = 20-40
```

**Reward Function R:**
```
R(s_t, a_t) = w_a · R_audio + w_v · R_visual + R_aux

R_audio  = cos_sim(Sylber(audio_gen), Sylber(audio_target))  ∈ [-1, 1]
R_visual = 1 - LVE_norm(mesh_gen, mesh_target)               ∈ [0, 1]
R_aux    = -λ_silence · 𝟙[no_syllable] - λ_smooth · ||a_t - a_{t-1}||² - λ_energy · ||a_t||²

Default weights: w_a = 0.6, w_v = 0.3, λ_silence = 1.0, λ_smooth = 0.01, λ_energy = 0.001
```

**Episode Structure:**
```
Horizon H = 50 steps (2 seconds at 25 Hz)
Discount γ = 0.99
Reset: articulators to neutral position
Termination: fixed horizon (no early termination)
```

### 6.4 Articulatory Model

Based on analysis of Anand et al. (13 DOF), VocalTractLab (33 DOF), and biomechanical literature:

| Articulator | DOF | Parameters | Biomechanical Basis |
|-------------|-----|------------|---------------------|
| Tongue Dorsum | 2 | X, Y position | Genioglossus posterior, styloglossus |
| Tongue Blade | 2 | X, Y position | Genioglossus middle, verticalis |
| Tongue Tip | 2 | X, Y position | Genioglossus anterior, superior longitudinal |
| Jaw | 3 | Open/close, protrusion, lateral | Masseter, pterygoids, digastric |
| Upper Lip | 2 | Protrusion, raise | Orbicularis oris superior, levator labii |
| Lower Lip | 2 | Protrusion, lower | Orbicularis oris inferior, depressor labii |
| Lip Corners | 2 | Spread/retract (L/R symmetric) | Zygomaticus, risorius |
| Velum | 1 | Open/close | Levator veli palatini, palatoglossus |
| Larynx | 2 | F0 (pitch), voicing amplitude | Cricothyroid, thyroarytenoid |
| **Total** | **18** | | |

Muscle dynamics follow a simplified Hill-type model:
```
F_muscle = F_max · activation · f_length(l) · f_velocity(v)

where:
  activation ∈ [0, 1] (from RL action, smoothed by τ_act = 10ms, τ_deact = 40ms)
  f_length = Gaussian centered at optimal muscle length
  f_velocity = Hill's force-velocity relationship
```

### 6.5 Technology Stack

| Component | Primary Choice | Rationale |
|-----------|---------------|-----------|
| **Language** | Python 3.11+ | Ecosystem compatibility |
| **RL Framework** | Stable-Baselines3 / CleanRL | Mature, Gymnasium-compatible, PPO+SAC |
| **Physics Engine** | MuJoCo (via MJX for GPU) | Fast, RL-ready, muscle-tendon model, open-source |
| **Alternate Physics** | NVIDIA Isaac Lab | If soft-body FEM fidelity needed |
| **Audio Decoder** | SPARC | State-of-the-art articulatory-to-speech |
| **Visual Rendering** | MuJoCo native + FLAME adapter | Direct from simulation mesh |
| **Audio Encoder** | Sylber + wav2vec 2.0 | Proven for articulatory RL reward |
| **Visual Encoder** | AV-HuBERT / lip-reading net | Cross-modal verification |
| **Experiment Tracking** | W&B + TensorBoard | Industry standard |
| **Config Management** | Hydra / OmegaConf | YAML-driven experiments |
| **Testing** | pytest | Standard |
| **Paper** | LaTeX (NeurIPS template) | Venue-appropriate |

### 6.6 Training Strategy

**Phase 1: Babbling (Exploration)**
- Random muscle activations
- Reward: any sound produced (binary)
- Purpose: learn basic articulator dynamics
- Episodes: ~5,000

**Phase 2: Vowel Imitation**
- Target: 5 cardinal vowels (/a/, /e/, /i/, /o/, /u/)
- Reward: audio similarity only
- Purpose: learn vowel articulation
- Episodes: ~10,000 per vowel

**Phase 3: Syllable Imitation**
- Target: CV and CVC syllables (20-50 syllables)
- Reward: audio + visual similarity
- Purpose: learn consonant-vowel transitions
- Episodes: ~25,000 per syllable

**Phase 4: Person-Specific Imitation**
- Target: specific speaker's utterances
- Reward: full multi-modal (audio + visual + commutative verification)
- Purpose: capture speaker-specific articulatory style
- Episodes: ~50,000+

---

## 7. Technical Considerations

### 7.1 Compute Requirements (Estimated)

| Phase | Envs | Episodes | Est. GPU Hours | Hardware |
|-------|------|----------|---------------|----------|
| Babbling | 64 | 5K | ~2 | 1x A100 |
| Vowels | 64 | 50K | ~20 | 1x A100 |
| Syllables | 64 | 500K | ~200 | 1-4x A100 |
| Person-specific | 64 | 500K+ | ~200+ | 1-4x A100 |
| **Total** | | | **~422** | |

### 7.2 Dependencies on External Models

| Model | Purpose | License | Size |
|-------|---------|---------|------|
| SPARC | Articulatory-to-speech | Apache 2.0 (expected) | ~100M params |
| Sylber | Syllabic embeddings | Research | ~50M params |
| wav2vec 2.0 | Audio encoding | MIT | 317M params |
| FLAME | 3D face mesh | Non-commercial research | ~10K params |
| AV-HuBERT | Audio-visual encoding | CC-BY-NC | 325M params |

### 7.3 Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Physics sim too slow for RL | Training infeasible | Use MuJoCo MJX (GPU) or reduce fidelity |
| Acoustic output unrealistic | Poor reward signal | Fall back to SPARC neural decoder |
| Reward hacking | Agent exploits reward without real speech | Commutative verification (lip-reading check) |
| Action space too large | Slow convergence | Start with 13 DOF (Anand et al.), scale up |
| Sim-to-real gap | Results not transferable | Document as future work, design modular sim |

---

## 8. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Syllable intelligibility | >80% correct human transcription | 20+ test syllables, 10+ human raters |
| Audio similarity | Sylber cosine sim > 0.85 | Automated on test set |
| Visual similarity | LVE < best baseline | Compared to SelfTalk/FaceFormer on VOCASET |
| Training efficiency | Converge within 500K episodes | Per curriculum phase |
| Reproducibility | Independent replication possible | Documented seeds, configs, hardware |
| Paper acceptance | Top-tier venue | NeurIPS/ICML/ICLR/ICRA/Interspeech |
| Code quality | Tests pass, documented | pytest coverage >80% |

---

## 9. References

### Core RL & Articulatory Control
1. Anand, A. et al. "Teaching Machines to Speak Using Articulatory Control" (2025). arXiv:2510.05619.
2. Shitov, D. et al. "Deep RL for Articulatory Synthesis in a Vowel-to-Vowel Imitation Task" (2023). Sensors 23(7):3437.
3. Nguyen, D.-P. et al. "RL Coupled with FEM for Facial Motion Learning" (2022). Computer Methods and Programs in Biomedicine.
4. Krug, P. et al. "Artificial Vocal Learning Guided by Phoneme Recognition and Visual Information" (2023). IEEE/ACM TASLP.

### Speech & Articulatory Models
5. Cho, C.J. et al. "SPARC: Coding Speech Through Vocal Tract Kinematics" (2024). arXiv:2406.12998.
6. Guenther, F. et al. "The DIVA Model of Speech Motor Control." Boston University.
7. Wang, Y. et al. "ArtSpeech" (2024). ACM MM 2024.
8. VocalTractLab. vocaltractlab.de.

### Lip/Facial Motion Learning
9. Hu, Y. et al. "Learning Realistic Lip Motions for Humanoid Face Robots" (2026). Science Robotics 11(110).
10. Peng, Z. et al. "SelfTalk: Self-Supervised Commutative Training for 3D Talking Faces" (2023). ACM MM 2023.
11. "CorrTalk: Correlation Between Hierarchical Speech and Facial Activity" (2023).

### 3D Face Animation
12. Fan, Y. et al. "FaceFormer: Speech-Driven 3D Facial Animation with Transformers" (2022). CVPR 2022.
13. Xing, J. et al. "CodeTalker: Discrete Motion Prior" (2023). CVPR 2023.
14. Liu, H. et al. "EMAGE: Unified Holistic Co-Speech Gesture Generation" (2024). CVPR 2024.

### Biomechanical Simulation
15. PKU MOCCA Lab. "MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters" (2023). SIGGRAPH Asia 2023.
16. ETH Zurich / Disney Research. "Learning a Generalized Physical Face Model From Data" (2024). SIGGRAPH 2024.
17. ArtiSynth. artisynth.org.

### Embodied AI & Multimodal
18. Driess, D. et al. "PaLM-E: An Embodied Multimodal Language Model" (2023).
19. Willett, F. et al. "A High-Performance Speech Neuroprosthesis" (2023). Nature.
20. MyoSuite. github.com/MyoHub/myosuite.

---

## 10. Open Questions

See Analysis Phase (PRD-C) for resolution of critical questions.
