# PRD-A: OpenJaw — Reinforcement Learning for Embodied Speech Imitation in Multi-Physics 3D Simulation

## Introduction/Overview

OpenJaw is a research project to develop a **mathematical RL (Reinforcement Learning) framework** for closed-loop embodied speech imitation, trained entirely within a **multi-physics 3D simulation** environment with audio-visual sensory feedback.

The core loop:
> **Audio + video perception** → **RL policy** → **control signals to a simulated multi-physics mouth** → **virtual sensory feedback (audio + vision)** → **imitation loss** → **policy improvement**

The system does **not** build a physical mouth, jaw, or tongue. Instead, it simulates them using multi-physics 3D simulation that produces realistic visual output and sound-wave propagation — so that virtual cameras and microphones can observe and provide feedback to the RL agent as it learns to imitate a specific person's speaking style.

**Deliverables:** A GitHub repository containing the RL model code, simulation environment, training pipeline, evaluation tools, and an accompanying scientific paper.

### Problem Statement

Current speech synthesis systems (TTS) bypass the physical process of speech production entirely — they map text or audio directly to waveforms without modeling the biomechanics of articulation. This limits:
- Understanding of speech motor control
- Ability to simulate realistic person-specific speaking styles (visual + auditory)
- Transfer to embodied systems (robots, prosthetics, neuroprosthetics)
- Scientific modeling of human speech acquisition

OpenJaw addresses this by treating speech production as a **motor control problem**: an RL agent must learn to coordinate mouth, jaw, lips, and tongue muscles to produce speech that matches a target person — both in how it sounds and how it looks.

## Goals

- Develop a mathematically rigorous RL framework for articulatory speech imitation
- Build a multi-physics 3D simulation of the human oral cavity (jaw, lips, tongue, teeth, palate, pharynx) that produces both visual and acoustic output
- Implement a closed-loop training pipeline: RL policy → simulation → virtual sensors → reward → policy update
- Demonstrate imitation of a specific person's speech patterns (audio + visual fidelity)
- Publish results as a peer-reviewed scientific paper with reproducible code
- Achieve state-of-the-art results on articulatory speech synthesis benchmarks

## User Stories

### US-001: RL Policy Architecture
**Description:** As a researcher, I want a well-defined RL policy architecture that maps multi-modal observations (audio + visual) to continuous articulator control signals, so that the agent can learn to produce speech through physical simulation.

**Acceptance Criteria:**
- [ ] Policy network accepts audio spectrogram + visual frame embeddings as input
- [ ] Policy outputs continuous control signals for all articulators (tongue 6-DOF, jaw 3-DOF, lips 4-6 DOF, velum, larynx)
- [ ] Supports PPO and SAC training algorithms
- [ ] Frame stacking for temporal context (minimum 15 frames)
- [ ] Action space bounded and normalized

### US-002: Multi-Physics Mouth Simulation
**Description:** As a researcher, I want a physically accurate 3D simulation of the human oral cavity that produces both visual deformation and acoustic wave propagation, so that the RL agent receives realistic sensory feedback.

**Acceptance Criteria:**
- [ ] 3D mesh model of jaw, lips, tongue, teeth, palate, pharynx
- [ ] Soft-body / FEM physics for tongue and lip deformation
- [ ] Rigid-body dynamics for jaw articulation
- [ ] Muscle-tendon actuation model for all articulators
- [ ] Acoustic output generated from the simulated vocal tract geometry
- [ ] Visual rendering of the mouth at each timestep
- [ ] Simulation runs at minimum 25 FPS for training feasibility

### US-003: Virtual Sensory Feedback System
**Description:** As a researcher, I want virtual cameras and microphones that observe the simulated mouth and produce audio-visual signals, so that the RL agent has a closed feedback loop for learning.

**Acceptance Criteria:**
- [ ] Virtual camera captures visual frames of the simulated mouth
- [ ] Virtual microphone captures acoustic output from the simulation
- [ ] Sensory signals are encoded into embeddings suitable for reward computation
- [ ] Latency between action and observation is physically consistent

### US-004: Imitation Reward Function
**Description:** As a researcher, I want a reward function that measures how closely the simulated speech matches a target person's speech (both audio and visual), so that the RL agent optimizes toward faithful imitation.

**Acceptance Criteria:**
- [ ] Audio similarity metric (e.g., cosine similarity of syllabic/phonetic embeddings)
- [ ] Visual similarity metric (e.g., lip-shape error, facial landmark distance)
- [ ] Combined multi-modal reward with configurable weighting
- [ ] Penalty for silence / no syllable production
- [ ] Reward is differentiable-friendly for potential hybrid optimization

### US-005: Training Pipeline
**Description:** As a researcher, I want an end-to-end training pipeline that loads target person reference data, runs RL episodes in simulation, and logs metrics, so that I can train and evaluate the model efficiently.

**Acceptance Criteria:**
- [ ] Load reference audio + video of target person
- [ ] Episode management (reset, step, terminate)
- [ ] Parallel environment support for faster training
- [ ] Logging of rewards, losses, articulator trajectories
- [ ] Checkpointing and resumable training
- [ ] Configurable hyperparameters via config files

### US-006: Evaluation and Benchmarking
**Description:** As a researcher, I want quantitative evaluation tools that measure the quality of generated speech against baselines, so that I can report results in the scientific paper.

**Acceptance Criteria:**
- [ ] Mel-cepstral distortion (MCD) metric
- [ ] Lip vertex error (LVE) metric
- [ ] Perceptual speech quality (PESQ or similar)
- [ ] Human evaluation protocol (MOS scoring)
- [ ] Comparison against baseline methods (VTL, SPARC, direct neural TTS)
- [ ] Visualization tools for articulator trajectories

### US-007: Scientific Paper
**Description:** As a researcher, I want the codebase structured to support reproducible experiments and paper-ready figures, so that results can be published in a peer-reviewed venue.

**Acceptance Criteria:**
- [ ] Experiment configs for all reported results
- [ ] Figure generation scripts
- [ ] LaTeX paper template with method, results, and discussion sections
- [ ] Reproducibility documentation (seeds, hardware specs, training time)

## Functional Requirements

- **FR-1:** The system must implement a PPO-based RL agent with continuous action space for articulatory control (13-33 dimensions)
- **FR-2:** The system must simulate a 3D oral cavity with deformable tongue, lips, and rigid jaw using FEM or equivalent physics
- **FR-3:** The system must generate acoustic output from the simulated vocal tract geometry at each timestep
- **FR-4:** The system must render visual frames of the simulated mouth at each timestep
- **FR-5:** The system must compute a multi-modal reward comparing generated audio-visual output against target reference
- **FR-6:** The system must support episode-based RL training with configurable episode length, action bounds, and reward weights
- **FR-7:** The system must support parallel simulation environments for training throughput
- **FR-8:** The system must log training metrics (reward, loss, articulator trajectories) to TensorBoard or W&B
- **FR-9:** The system must support loading reference audio + video data of a target speaker
- **FR-10:** The system must implement at least two RL algorithms (PPO and SAC) for comparison
- **FR-11:** The system must provide evaluation scripts computing MCD, LVE, and perceptual quality metrics
- **FR-12:** The system must support checkpointing and resumable training

## Non-Goals (Out of Scope)

- Building or controlling a physical robot mouth (hardware)
- Real-time inference / deployment as a production TTS system
- Text-to-speech (input is reference audio/video, not text)
- Emotional expression synthesis beyond speech articulation
- Full-body gesture or head-motion generation
- Language-specific phoneme engineering (the model learns from observation, not linguistic rules)
- Cloud deployment or web-based interface

## Design Considerations

- **Simulation Fidelity vs. Speed:** The simulation must be physically accurate enough for meaningful RL training but fast enough for millions of timesteps. GPU-accelerated physics (Isaac Lab, Warp) may be necessary.
- **Modular Architecture:** The RL policy, simulation environment, reward function, and evaluation tools should be decoupled for independent development and ablation studies.
- **Reproducibility:** All experiments must be reproducible from config files with fixed seeds.

## Technical Considerations

### Candidate Physics Engines
| Engine | Strengths | Considerations |
|--------|-----------|----------------|
| NVIDIA Isaac Lab | GPU-accelerated FEM, PhysX 5, PyTorch integration | New framework, limited articulatory models |
| MuJoCo | Fast, RL-ready, Gymnasium API, muscle-tendon | No native orofacial model, rigid tendons |
| ArtiSynth | Purpose-built vocal tract, FEM + mass-spring | Java, not GPU-accelerated |
| SOFA | Full FEM, medical sim | Heavy, not RL-integrated |

### Candidate Audio Synthesis
| Method | Strengths | Considerations |
|--------|-----------|----------------|
| SPARC | Neural articulatory-to-speech, generalizes | Requires pre-trained decoder |
| VocalTractLab | 33-parameter classical synthesizer | Geometric model, not full FEM |
| FEM Acoustics | Physically accurate 3D wave propagation | Computationally expensive |

### Candidate RL Algorithms
| Algorithm | Type | Why |
|-----------|------|-----|
| PPO | On-policy | Proven for articulatory control (Anand et al. 2025) |
| SAC | Off-policy | Better sample efficiency, entropy exploration |
| TD3 | Off-policy | Used in FEM facial control (Nguyen et al. 2022) |
| Model-Based RL | Hybrid | Sample efficient when sim is expensive (Shitov et al. 2023) |

### Key Perception Models
- **Sylber:** Self-supervised syllabic embeddings for audio reward
- **wav2vec 2.0:** Speech encoder for audio feature extraction
- **FLAME:** 3D morphable face model (5023 vertices) for visual representation
- **Lip-reading networks:** Visual verification feedback (SelfTalk paradigm)

## Success Metrics

- RL agent produces intelligible syllables (>80% correct human transcription on 20+ test syllables)
- Audio cosine similarity to target > 0.85 (Sylber embedding space)
- Lip vertex error < state-of-the-art baselines (SelfTalk, FaceFormer)
- Paper accepted at a top-tier venue (NeurIPS, ICML, ICLR, ICRA, or Speech/Audio venue)
- Codebase reproducible: independent researcher can replicate key results within documented compute budget

## References

### Core RL & Articulatory Control
1. Anand et al. "Teaching Machines to Speak Using Articulatory Control" (2025) — arXiv:2510.05619
2. Shitov et al. "Deep RL for Articulatory Synthesis in a Vowel-to-Vowel Imitation Task" (2023) — Sensors 23(7):3437
3. Nguyen et al. "RL Coupled with FEM for Facial Motion Learning" (2022) — Computer Methods and Programs in Biomedicine
4. Krug et al. "Artificial Vocal Learning Guided by Phoneme Recognition and Visual Information" (2023) — IEEE/ACM TASLP

### Speech & Articulatory Models
5. Cho et al. "SPARC: Coding Speech Through Vocal Tract Kinematics" (2024) — arXiv:2406.12998
6. Guenther et al. "The DIVA Model of Speech Motor Control" — Boston University
7. Wang et al. "ArtSpeech" (2024) — ACM MM 2024
8. VocalTractLab — vocaltractlab.de

### Lip/Facial Motion Learning
9. Hu et al. "Learning Realistic Lip Motions for Humanoid Face Robots" (2026) — Science Robotics 11(110)
10. "Mirror-Taught Bionic Face Lip-Syncs Speech and Song Across Languages" (2026)
11. Peng et al. "SelfTalk: Self-Supervised Commutative Training for 3D Talking Faces" (2023) — ACM MM 2023
12. "CorrTalk: Correlation Between Hierarchical Speech and Facial Activity" (2023)

### 3D Face Animation
13. Fan et al. "FaceFormer: Speech-Driven 3D Facial Animation with Transformers" (2022) — CVPR 2022
14. Xing et al. "CodeTalker: Discrete Motion Prior" (2023) — CVPR 2023
15. Liu et al. "EMAGE: Unified Holistic Co-Speech Gesture Generation" (2024) — CVPR 2024

### Biomechanical Simulation
16. "MuscleVAE: Model-Based Controllers of Muscle-Actuated Characters" (2023) — SIGGRAPH Asia 2023
17. "Learning a Generalized Physical Face Model From Data" (2024) — SIGGRAPH 2024 (ETH/Disney)
18. ArtiSynth — artisynth.org

### Embodied AI & Multimodal
19. Driess et al. "PaLM-E: An Embodied Multimodal Language Model" (2023)
20. Willett et al. "A High-Performance Speech Neuroprosthesis" (2023) — Nature

## Open Questions

1. Which physics engine should be primary? (Isaac Lab for GPU speed vs. ArtiSynth for vocal tract fidelity)
2. Should acoustic output come from physical simulation (FEM acoustics) or neural decoder (SPARC)?
3. What is the minimum simulation fidelity needed for meaningful RL training?
4. How many articulatory DOFs are needed? (13 minimal vs. 33 full VTL parameterization)
5. Should the model train syllable-by-syllable or on continuous speech?
6. What reference dataset(s) of target speaker audio + video should be used?
7. How should the visual and audio rewards be weighted relative to each other?
8. What is the target compute budget for training?
9. Should the paper target a robotics venue (ICRA), ML venue (NeurIPS/ICML), or speech venue (Interspeech)?
10. Is sim-to-real transfer (to a physical robot) a future consideration that should influence architecture?
