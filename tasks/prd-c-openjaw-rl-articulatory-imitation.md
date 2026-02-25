# PRD-C: OpenJaw — Mathematical Framework and Neural Architecture for RL-Based Visual-Acoustic Articulatory Speech Imitation

> **Version:** PRD-C (Final — Clarified Scope)
> **Date:** 2026-02-24
> **Status:** Approved for Dev-Plan

---

## 1. Introduction/Overview

### What OpenJaw Is

OpenJaw is a **mathematical and architectural framework** for closed-loop reinforcement learning-based embodied speech imitation. The project delivers:

1. **A rigorous mathematical model** — the MDP formulation, state/action/observation spaces, reward structure, and transition dynamics for articulatory speech control
2. **A neural network architecture** — the RL policy network, perception encoders, and reward modules designed for visual-acoustic articulatory imitation
3. **A reference implementation** — code that demonstrates the architecture in a MuJoCo-based simulation with SPARC audio decoding
4. **A scientific paper** — targeting a top ML venue (NeurIPS/ICML/ICLR), presenting the framework, architecture analysis, and proof-of-concept results
5. **Methods analysis** — rigorous justification of all design decisions (physics engine, audio method, DOF count, scope, data, compute) as a contribution to the field

### What OpenJaw Is NOT

This is not a production speech synthesis system. The goal is to **define and validate the architecture and mathematical foundation** upon which large-scale RL training would later be built. The code demonstrates feasibility on a single consumer GPU; scaling is future work.

### The Core Loop

```
Reference Audio/Video (Target Speaker)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  Observation  ┌──────────────┐  Actions   ┌──────────┐  │
│  Encoder   ──▶│  RL Policy   │──────────▶│ MuJoCo   │  │
│  (wav2vec +   │  (PPO, MLP   │  13-DOF    │ Mouth    │  │
│   visual)     │   Actor-     │  muscle    │ Sim      │  │
│               │   Critic)    │  activat.  │          │  │
│               └──────▲───────┘            └────┬─────┘  │
│                      │                         │        │
│               ┌──────┴───────┐           ┌─────▼─────┐  │
│               │   Reward     │◀──────────│  SPARC    │  │
│               │   Module     │  audio    │  Decoder  │  │
│               │  (Sylber     │◀──────────│  + Mesh   │  │
│               │   + LVE)     │  visual   │  Render   │  │
│               └──────────────┘           └───────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Scientific Contribution

OpenJaw is the **first unified mathematical framework** that jointly models:
- RL-driven articulatory motor control (13 DOF)
- Multi-physics oral cavity dynamics (MuJoCo rigid + constrained soft body)
- Dual-modality output (SPARC audio + mesh visual)
- Dual-modality perception and reward (Sylber audio + LVE visual)
- Closed-loop sensory feedback for policy improvement

The paper positions this as a **foundational architecture paper** for the ML community, with methods analysis justifying every design choice.

---

## 2. Design Decisions (Methods Analysis)

All 7 critical design decisions are justified here and will form a dedicated **Methods Analysis** section in the paper.

### D1: Physics Engine — MuJoCo + MJX

**Decision:** MuJoCo with MJX GPU acceleration

**Analysis:**
| Engine | Speed (steps/s) | RL Integration | Soft Body | GPU | Articulatory Models |
|--------|-----------------|----------------|-----------|-----|-------------------|
| MuJoCo + MJX | ~1M+ | Native Gymnasium | Constrained | Yes (MJX/XLA) | Muscle-tendon, custom |
| Isaac Lab | ~100K+ | Good | FEM (PhysX 5) | Yes | None existing |
| ArtiSynth | ~1K | None | FEM + mass-spring | No | Vocal tract built-in |
| SOFA | ~500 | Poor | Full FEM | Limited | None |

**Justification:** MuJoCo provides the fastest RL training loop, proven Gymnasium integration, and muscle-tendon actuation. MJX enables GPU-parallel environments critical for consumer GPU compute budgets. While MuJoCo lacks native FEM soft-body, the tongue and lips can be modeled using tendon-driven constraints and composite bodies sufficient for the 13-DOF control space. ArtiSynth's vocal tract fidelity is higher but 1000x slower — unsuitable for RL. The paper will discuss this trade-off as **simulation fidelity vs. training throughput**, arguing that RL convergence requires millions of steps, making speed the binding constraint.

### D2: Audio Output — SPARC Neural Decoder

**Decision:** SPARC (Speech Articulatory Coding) for articulatory-to-audio mapping

**Analysis:**
| Method | Fidelity | Speed | Physical Grounding | RL Compatibility |
|--------|----------|-------|-------------------|-----------------|
| SPARC | High (neural) | Fast (~1ms) | Learned mapping | Excellent |
| FEM Acoustics | Highest | Very slow (~100ms) | First-principles | Poor (too slow) |
| VocalTractLab | Medium | Medium (~10ms) | Geometric | Moderate |

**Justification:** SPARC maps articulatory kinematic trajectories directly to speech waveforms via a pre-trained neural encoder-decoder. It generalizes to unseen speakers and produces high-quality audio at negligible latency. While not first-principles physics, SPARC's articulatory grounding (trained on real electromagnetic articulography data) provides a physically meaningful mapping. FEM acoustics, though more principled, is 100x too slow for RL training on consumer hardware. The paper frames SPARC as a **differentiable acoustic proxy** that preserves the articulatory control structure while enabling tractable training.

### D3: Target Venue — ML (NeurIPS/ICML/ICLR)

**Decision:** Target NeurIPS or ICML

**Justification:** The contribution is fundamentally a **novel RL architecture and mathematical framework**, not an empirical robotics demo or a speech quality benchmark. ML venues value: (1) novel problem formulations, (2) principled architecture design, (3) rigorous ablations, (4) potential for broad impact. OpenJaw introduces a new problem setting (closed-loop visual-acoustic articulatory RL) to the ML community. The paper will be structured as a framework paper with proof-of-concept experiments, following the template of PaLM-E (2023) and similar cross-domain ML contributions.

### D4: Scope — Syllable-Level Imitation

**Decision:** Syllable-level imitation as proof-of-concept

**Justification:** Anand et al. (2025) demonstrated syllable-level articulatory RL with 13 DOF and PPO, achieving 0.85+ cosine similarity on 6 syllables. OpenJaw extends this baseline with: (1) physics simulation instead of direct SPARC control, (2) visual output and feedback, (3) person-specific imitation loss. Syllable-level scope keeps training feasible on consumer GPU (~25K episodes per syllable) while demonstrating the full architecture. Word-level and continuous speech are positioned as future work enabled by the framework.

### D5: Reference Data — VOCASET + Custom Recording

**Decision:** VOCASET for quantitative benchmarking, custom recording for person-specific demo

**Justification:** VOCASET (480 sequences, 12 subjects, FLAME topology, 60 FPS) is the standard benchmark used by SelfTalk, FaceFormer, and CodeTalker — enabling direct comparison. A custom recording of a target speaker demonstrates the person-specific imitation capability that distinguishes OpenJaw from prior work. The custom recording needs: high-quality audio (48kHz), frontal video (1080p, 60fps), and ideally FLAME parameter extraction via existing tools.

### D6: Compute Budget — Consumer GPU (~100 GPU-hours)

**Decision:** Single RTX 3090/4090, ~100 GPU-hours total

**Justification:** This constraint shapes the architecture toward sample efficiency. Implications:
- MuJoCo MJX parallelization essential (64+ envs on single GPU)
- PPO with small MLP networks (not large transformers)
- Sylber/wav2vec encoders frozen (no fine-tuning of perception models)
- Syllable-level episodes (50 steps = 2s) not long sequences
- ~25K episodes per syllable × 20 syllables = 500K episodes feasible

The paper frames this as a feature: **"We demonstrate that the proposed architecture achieves meaningful speech imitation on consumer hardware, establishing accessibility as a design principle."**

### D7: Articulatory DOF — 13 (Minimal, Anand et al.)

**Decision:** 13 DOF following Anand et al. (2025)

| DOF | Articulator | Parameters |
|-----|------------|------------|
| 1-2 | Tongue Dorsum | X, Y velocity |
| 3-4 | Tongue Blade | X, Y velocity |
| 5-6 | Tongue Tip | X, Y velocity |
| 7-8 | Lower Incisor (Jaw) | X, Y velocity |
| 9-10 | Upper Lip | X, Y velocity |
| 11-12 | Lower Lip | X, Y velocity |
| 13 | Vocal Loudness | Scalar |

**Justification:** 13 DOF is the minimal proven set for syllable-level speech production. Starting minimal: (1) reduces action space dimensionality → faster RL convergence, (2) enables direct comparison with Anand et al., (3) fits consumer GPU budget. The paper includes analysis of DOF sufficiency and discusses scaling to 18-33 DOF as future work. The architecture is designed to be DOF-agnostic — increasing DOF requires only config changes, not architectural modification.

---

## 3. Goals (Refined)

- **G1:** Formalize the complete MDP for visual-acoustic articulatory imitation (state, action, observation, reward, dynamics)
- **G2:** Design the neural network architecture: policy network, perception encoders, reward modules
- **G3:** Implement the architecture in MuJoCo + SPARC with Gymnasium interface
- **G4:** Demonstrate proof-of-concept: RL agent produces recognizable syllables via closed-loop training
- **G5:** Provide rigorous methods analysis justifying all 7 design decisions
- **G6:** Ablation studies: audio-only vs. visual-only vs. combined reward; 13 vs. 18 DOF
- **G7:** Write and submit paper to NeurIPS/ICML/ICLR
- **G8:** Release reproducible open-source codebase

---

## 4. User Stories (Refined for Architecture Focus)

### US-001: Mathematical Framework (MDP Formulation)
**Description:** As a researcher, I want a rigorous MDP formulation documented in code and paper, so that the framework has solid mathematical foundations.

**Acceptance Criteria:**
- [ ] State space S ∈ ℝ^{39×15} (13 positions + 13 velocities + 13 previous actions, frame-stacked 15)
- [ ] Action space A ∈ [-0.5, 0.5]^13 (articulator velocities + loudness)
- [ ] Observation space O: frame-stacked state + target embedding
- [ ] Reward R = w_a·cos_sim(Sylber_gen, Sylber_target) + w_v·(1 - LVE_norm) + R_aux
- [ ] Transition T: MuJoCo physics step with 20 substeps per control step
- [ ] Episode: H=50 steps, γ=0.99, reset to neutral
- [ ] All formulations in LaTeX in the paper and as Python dataclasses in code

### US-002: Policy Network Architecture
**Description:** As a researcher, I want a well-designed actor-critic network architecture for the 13-DOF continuous control problem, documented as a paper contribution.

**Acceptance Criteria:**
- [ ] Actor: MLP (observation_dim → 256 → 256 → 13), tanh output
- [ ] Critic: MLP (observation_dim → 256 → 256 → 1)
- [ ] Standard deviation: learnable or scheduled (0.7 initial → 0.05 minimum)
- [ ] Orthogonal weight initialization
- [ ] Architecture diagram in paper
- [ ] Parameter count documented (<1M total)

### US-003: MuJoCo Oral Cavity Model
**Description:** As a researcher, I want a MuJoCo XML model of the oral cavity with 13-DOF tendon-driven actuation, so that the RL agent has a physically grounded environment.

**Acceptance Criteria:**
- [ ] MuJoCo XML defining jaw, tongue (3 segments), lips (upper/lower), teeth, palate
- [ ] Tendon-driven actuation for all 13 DOFs
- [ ] Anatomically motivated geometry (sagittal cross-section, extruded to 3D)
- [ ] Contact pairs: tongue-palate, tongue-teeth, lip-lip
- [ ] Runs at >1000 steps/second on single GPU
- [ ] Gymnasium Env wrapper: reset(), step(), render()

### US-004: SPARC Audio Integration
**Description:** As a researcher, I want the articulatory state to be decoded to audio via SPARC at each control step, so that the RL agent receives acoustic feedback.

**Acceptance Criteria:**
- [ ] SPARC model loaded and frozen (no fine-tuning)
- [ ] Input: 13-DOF articulatory trajectory (current + history)
- [ ] Output: audio waveform at 16kHz
- [ ] Inference < 5ms per step on consumer GPU
- [ ] Validated: known articulatory trajectories produce expected audio

### US-005: Visual Rendering Pipeline
**Description:** As a researcher, I want the MuJoCo simulation to produce visual frames compatible with evaluation metrics, so that visual feedback is part of the closed loop.

**Acceptance Criteria:**
- [ ] MuJoCo native renderer produces 256×256 mouth images
- [ ] Mesh vertices extractable in FLAME-compatible format for LVE computation
- [ ] Frontal camera view
- [ ] 25 FPS (one frame per control step)

### US-006: Perception Encoder Module
**Description:** As a researcher, I want frozen pre-trained encoders that convert audio and visual output to embeddings for reward computation.

**Acceptance Criteria:**
- [ ] Sylber encoder: audio → syllabic embedding (frozen)
- [ ] wav2vec 2.0: audio → frame-level features (frozen, for observation)
- [ ] Visual encoder: mesh vertices → lip region features (for LVE)
- [ ] All encoders run on same GPU as training
- [ ] Total encoder overhead < 10ms per step

### US-007: Multi-Modal Reward Module
**Description:** As a researcher, I want the reward function implemented as a modular, configurable component with ablation support.

**Acceptance Criteria:**
- [ ] R_audio = cos_sim(Sylber(gen), Sylber(target))
- [ ] R_visual = 1 - normalized_LVE(mesh_gen, mesh_target)
- [ ] R_combined = w_a * R_audio + w_v * R_visual
- [ ] R_aux = -λ_silence * no_syllable - λ_smooth * ||Δa||² - λ_energy * ||a||²
- [ ] Weights configurable via YAML
- [ ] Can disable audio or visual independently (for ablations)

### US-008: Training Pipeline
**Description:** As a researcher, I want a training script that runs PPO on the MuJoCo environment with all modules connected.

**Acceptance Criteria:**
- [ ] PPO via Stable-Baselines3 or CleanRL
- [ ] Vectorized environments (16-64 parallel)
- [ ] Curriculum: babbling (1K eps) → vowels (5K eps) → syllables (25K eps)
- [ ] TensorBoard logging: reward, episode length, articulator stats
- [ ] Checkpointing every 1K episodes
- [ ] Total training < 100 GPU-hours on RTX 3090/4090
- [ ] Reproducible with fixed seed

### US-009: Evaluation and Ablation Suite
**Description:** As a researcher, I want evaluation scripts that produce paper-ready metrics and figures.

**Acceptance Criteria:**
- [ ] MCD (Mel-Cepstral Distortion) computation
- [ ] LVE (Lip Vertex Error) computation
- [ ] CER (Character Error Rate) via ASR
- [ ] Sylber cosine similarity
- [ ] Ablation: audio-only reward vs. visual-only vs. combined
- [ ] Ablation: with/without curriculum
- [ ] Articulator trajectory plots
- [ ] Reward curve plots
- [ ] Side-by-side audio comparison

### US-010: Scientific Paper
**Description:** As a researcher, I want a complete paper draft with all sections.

**Acceptance Criteria:**
- [ ] Abstract: problem, approach, key result
- [ ] Introduction: motivation, gap, contribution
- [ ] Related Work: RL for articulation, talking heads, biomechanical sim
- [ ] Methods: MDP formulation, architecture, all 7 design decisions analyzed
- [ ] Experiments: proof-of-concept results, ablations
- [ ] Discussion: limitations, future work (scaling DOF, compute, continuous speech)
- [ ] Figures: architecture diagram, reward curves, articulator trajectories, audio spectrograms
- [ ] NeurIPS/ICML LaTeX format

---

## 5. Functional Requirements (Refined)

### Core Architecture (Paper Contribution)
- **FR-1:** Complete MDP formulation as Python dataclasses and LaTeX equations
- **FR-2:** Policy network (MLP actor-critic, <1M parameters) for 13-DOF continuous control
- **FR-3:** MuJoCo oral cavity model with anatomically motivated geometry and tendon actuation
- **FR-4:** SPARC integration for articulatory-to-audio decoding (frozen, inference only)
- **FR-5:** Visual rendering from MuJoCo mesh for lip vertex extraction
- **FR-6:** Sylber + wav2vec perception encoders (frozen, inference only)
- **FR-7:** Multi-modal reward: R = w_a·R_audio + w_v·R_visual + R_aux

### Training & Evaluation
- **FR-8:** PPO training pipeline with vectorized MuJoCo environments
- **FR-9:** Curriculum scheduler (babbling → vowels → syllables)
- **FR-10:** Evaluation metrics: MCD, LVE, CER, Sylber similarity
- **FR-11:** Ablation framework: toggle reward components, DOF count
- **FR-12:** TensorBoard logging and checkpointing
- **FR-13:** VOCASET data loader for benchmarking
- **FR-14:** Custom speaker data loader (audio + video → embeddings)

### Paper Artifacts
- **FR-15:** LaTeX paper in NeurIPS/ICML format
- **FR-16:** Figure generation scripts (architecture, results, ablations)
- **FR-17:** Reproducibility documentation (seeds, configs, hardware specs)

---

## 6. Non-Goals (Out of Scope)

- Physical robot construction or sim-to-real transfer
- Production TTS system or real-time inference
- Text-to-speech (input is reference audio/video)
- Training large transformer policies (consumer GPU constraint)
- Fine-tuning SPARC, Sylber, or wav2vec (frozen inference only)
- FEM soft-body simulation (MuJoCo constrained bodies sufficient for 13 DOF)
- Multi-speaker simultaneous training
- Emotional expression, full-body gesture, head motion
- Cloud deployment, web interface, or API

---

## 7. Architecture (Detailed)

### 7.1 Neural Network Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    OPENJAW NEURAL ARCHITECTURE                    │
│                                                                   │
│  PERCEPTION (Frozen)          POLICY (Trained)     ENVIRONMENT    │
│  ┌─────────────────┐         ┌──────────────┐     ┌──────────┐  │
│  │ wav2vec 2.0     │         │              │     │          │  │
│  │ (audio → 768d)  │──┐      │  Actor       │     │ MuJoCo   │  │
│  └─────────────────┘  │      │  MLP         │     │ Oral     │  │
│  ┌─────────────────┐  │ obs  │  256→256→13  │ act │ Cavity   │  │
│  │ Sylber          │  ├─────▶│  (tanh)      │────▶│          │  │
│  │ (audio → emb)   │  │      │              │     │ 13-DOF   │  │
│  └─────────────────┘  │      │  Critic      │     │ tendon   │  │
│  ┌─────────────────┐  │      │  MLP         │     │ driven   │  │
│  │ Visual Encoder  │──┘      │  256→256→1   │     │          │  │
│  │ (mesh → feat)   │         │              │     └────┬─────┘  │
│  └─────────────────┘         └──────────────┘          │        │
│                                     ▲                   │        │
│  REWARD (Frozen encoders)           │ reward            │        │
│  ┌─────────────────┐                │              ┌────▼─────┐  │
│  │ R_audio:        │                │              │ SPARC    │  │
│  │ Sylber cos_sim  │────────────────┤              │ (artic → │  │
│  │                 │                │              │  audio)  │  │
│  │ R_visual:       │                │              └────┬─────┘  │
│  │ LVE on mesh     │────────────────┘              ┌────▼─────┐  │
│  │                 │                               │ Mesh     │  │
│  │ R_aux:          │                               │ Renderer │  │
│  │ silence+smooth  │                               └──────────┘  │
│  └─────────────────┘                                             │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Observation Vector Construction

```python
# Per-step observation (before frame stacking)
obs_t = concat([
    articulator_positions,   # 13 dims (from MuJoCo)
    articulator_velocities,  # 13 dims (from MuJoCo)
    previous_action,         # 13 dims
    target_syllable_embed,   # d_sylber dims (Sylber embedding of target)
])
# d_obs_frame = 39 + d_sylber

# Frame-stacked observation (K=15 frames)
obs_stacked = stack([obs_{t-14}, ..., obs_{t-1}, obs_t])  # shape: (15, d_obs_frame)
# Flattened: 15 * d_obs_frame dimensions
```

### 7.3 Reward Function (Formal)

```
R(s_t, a_t) = w_a · R_audio(s_t) + w_v · R_visual(s_t) + R_aux(a_t, a_{t-1})

where:
  R_audio  = cos(Sylber(SPARC(traj_t)), Sylber(audio_target))         ∈ [-1, 1]
  R_visual = 1 - ||v_lips^gen - v_lips^target||₂ / v_norm             ∈ [0, 1]
  R_aux    = -1.0 · 𝟙[no_syllable_detected]
             -0.01 · ||a_t - a_{t-1}||²
             -0.001 · ||a_t||²

Default: w_a = 0.7, w_v = 0.3
```

### 7.4 Curriculum Schedule

| Phase | Target | Episodes | Reward Components | Expected Outcome |
|-------|--------|----------|------------------|-----------------|
| 0: Babbling | Any sound | 1,000 | Binary (sound/silence) | Learn articulator dynamics |
| 1: Vowels | /a/, /e/, /i/, /o/, /u/ | 5,000 each | R_audio only | Learn vowel configurations |
| 2: CV Syllables | /ba/, /da/, /ka/, /ma/, etc. | 25,000 each | R_audio + R_visual | Learn consonant transitions |
| 3: Person-specific | Target speaker syllables | 25,000 each | Full R | Speaker style imitation |

### 7.5 Repository Structure

```
OpenJaw/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   ├── ppo_syllable.yaml
│   ├── curriculum.yaml
│   └── ablations/
│       ├── audio_only.yaml
│       ├── visual_only.yaml
│       └── no_curriculum.yaml
├── openjaw/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── mdp.py              # MDP dataclasses (S, A, O, R formal defs)
│   │   └── types.py            # Shared types
│   ├── env/
│   │   ├── __init__.py
│   │   ├── mouth_env.py        # Gymnasium env (main entry)
│   │   ├── oral_cavity.py      # MuJoCo model loader
│   │   ├── articulators.py     # 13-DOF articulator definitions
│   │   └── assets/
│   │       └── mouth.xml       # MuJoCo XML model
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── sparc_decoder.py    # SPARC wrapper
│   │   └── glottal.py          # Glottal source (if needed)
│   ├── visual/
│   │   ├── __init__.py
│   │   ├── renderer.py         # MuJoCo rendering
│   │   └── flame_adapter.py    # FLAME mesh mapping
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── sylber.py           # Sylber encoder wrapper
│   │   ├── wav2vec.py          # wav2vec 2.0 wrapper
│   │   └── lip_reader.py       # Visual encoder
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── audio_reward.py     # Sylber cosine similarity
│   │   ├── visual_reward.py    # Lip vertex error
│   │   ├── combined.py         # Multi-modal reward
│   │   └── auxiliary.py        # Penalties
│   ├── policy/
│   │   ├── __init__.py
│   │   └── networks.py         # MLP Actor-Critic
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # PPO training loop
│   │   ├── curriculum.py       # Phase scheduler
│   │   └── logger.py           # TensorBoard/W&B
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vocaset.py          # VOCASET loader
│   │   ├── custom_speaker.py   # Custom recording loader
│   │   └── preprocessing.py    # Feature extraction
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py          # MCD, LVE, CER
│       ├── ablations.py        # Ablation runner
│       └── visualization.py    # Plots and figures
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── figures/
│   └── scripts/
│       └── generate_figures.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   ├── test_mdp.py
│   ├── test_env.py
│   ├── test_audio.py
│   ├── test_visual.py
│   ├── test_reward.py
│   ├── test_perception.py
│   └── test_training.py
├── Plan/                       # Original planning docs
└── tasks/                      # PRD docs
```

---

## 8. Technical Considerations

### 8.1 Compute Constraints (RTX 3090/4090)

| Resource | RTX 3090 | RTX 4090 |
|----------|----------|----------|
| VRAM | 24 GB | 24 GB |
| CUDA Cores | 10,496 | 16,384 |
| FP32 TFLOPS | 35.6 | 82.6 |

**Budget allocation (~100 GPU-hours):**
- Env development & debugging: ~10 hrs
- Babbling + Vowel training: ~10 hrs
- Syllable training (20 syllables): ~60 hrs
- Evaluation + ablations: ~20 hrs

**Memory budget per parallel env:**
- MuJoCo state: ~1 KB
- SPARC inference: ~400 MB (shared)
- Sylber inference: ~200 MB (shared)
- wav2vec inference: ~1.2 GB (shared)
- Per-env overhead: ~5 MB
- Total: ~2 GB shared + 5 MB × N_envs
- With 64 envs: ~2.3 GB → fits in 24 GB with room for policy training

### 8.2 External Dependencies

| Dependency | Purpose | License | Install |
|------------|---------|---------|---------|
| MuJoCo 3.x | Physics simulation | Apache 2.0 | `pip install mujoco` |
| MJX | GPU MuJoCo | Apache 2.0 | Included with MuJoCo |
| Stable-Baselines3 | PPO implementation | MIT | `pip install stable-baselines3` |
| SPARC | Audio decoder | Research | GitHub clone |
| Sylber | Audio embeddings | Research | GitHub clone |
| wav2vec 2.0 | Audio features | MIT | `pip install transformers` |
| FLAME | Face mesh model | Non-commercial | Download from site |
| Gymnasium | Env interface | MIT | `pip install gymnasium` |
| PyTorch 2.x | Training | BSD | `pip install torch` |
| Hydra | Config | MIT | `pip install hydra-core` |

### 8.3 Key Risks (Updated for Scope)

| Risk | Impact | Mitigation |
|------|--------|------------|
| MuJoCo mouth model insufficient | Unrealistic dynamics | Start simple (2D sagittal), validate against ArtiSynth |
| SPARC doesn't accept MuJoCo state format | No audio output | Build adapter layer; SPARC takes articulatory positions |
| Consumer GPU too slow for 64 envs | Training infeasible | Reduce to 16 envs; extend training time |
| Reward hacking | Agent exploits reward | Commutative visual check; inspect trajectories |
| Paper contribution unclear | Rejection | Strong methods analysis section; position as framework paper |

---

## 9. Success Metrics (Refined)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Architecture completeness | All modules implemented and connected | Pytest: full pipeline test passes |
| MDP formalization | Complete and mathematically consistent | Paper review: no gaps in formulation |
| Syllable intelligibility | >50% correct on 10+ syllables | Human transcription (5+ raters) |
| Audio similarity | Sylber cos_sim > 0.7 | Automated metric on test set |
| Visual plausibility | LVE decreases during training | Training curve analysis |
| Methods analysis | All 7 decisions rigorously justified | Paper section completeness |
| Ablation coverage | Audio-only, visual-only, combined compared | Automated experiment suite |
| Reproducibility | Another researcher can run training | README test by independent person |
| Compute budget | Total ≤ 100 GPU-hours on RTX 3090/4090 | Wall-clock tracking |
| Paper quality | Submittable to NeurIPS/ICML | Self-review checklist |

---

## 10. References

1. Anand, A. et al. "Teaching Machines to Speak Using Articulatory Control" (2025). arXiv:2510.05619.
2. Shitov, D. et al. "Deep RL for Articulatory Synthesis in a Vowel-to-Vowel Imitation Task" (2023). Sensors 23(7):3437.
3. Nguyen, D.-P. et al. "RL Coupled with FEM for Facial Motion Learning" (2022). Comp. Methods Programs Biomed.
4. Krug, P. et al. "Artificial Vocal Learning Guided by Phoneme Recognition" (2023). IEEE/ACM TASLP.
5. Cho, C.J. et al. "SPARC: Coding Speech Through Vocal Tract Kinematics" (2024). arXiv:2406.12998.
6. Guenther, F. et al. "The DIVA Model of Speech Motor Control." Boston University.
7. Hu, Y. et al. "Learning Realistic Lip Motions for Humanoid Face Robots" (2026). Science Robotics.
8. Peng, Z. et al. "SelfTalk" (2023). ACM MM 2023.
9. Fan, Y. et al. "FaceFormer" (2022). CVPR 2022.
10. Xing, J. et al. "CodeTalker" (2023). CVPR 2023.
11. PKU MOCCA Lab. "MuscleVAE" (2023). SIGGRAPH Asia 2023.
12. ETH/Disney. "Generalized Physical Face Model" (2024). SIGGRAPH 2024.
13. Liu, H. et al. "EMAGE" (2024). CVPR 2024.
14. Willett, F. et al. "Speech Neuroprosthesis" (2023). Nature.
15. Wang, Y. et al. "ArtSpeech" (2024). ACM MM 2024.
16. Driess, D. et al. "PaLM-E" (2023).
17. VocalTractLab. vocaltractlab.de.
18. ArtiSynth. artisynth.org.
19. MyoSuite. github.com/MyoHub/myosuite.
20. FLAME. flame.is.tue.mpg.de.
