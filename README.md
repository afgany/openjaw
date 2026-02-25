# OpenJaw

**Mathematical RL Framework for Closed-Loop Visual-Acoustic Articulatory Speech Imitation**

📄 **[Read the Paper (PDF)](paper/main.pdf)**

OpenJaw is a research framework that formulates speech production as a reinforcement learning problem: an agent learns to control 13 articulatory degrees of freedom (tongue, jaw, lips) in a MuJoCo physics simulation, receiving audio (via SPARC decoder) and visual (via mesh rendering) feedback to imitate a target speaker's speech patterns.

Targeting NeurIPS/ICML/ICLR. See the [paper](paper/main.tex) for the full technical description.

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/your-org/openjaw.git
cd openjaw

# Install (Python 3.12+)
pip install -e ".[dev]"

# Verify
pytest tests/ -v
```

### Run Tests (201 tests)

```bash
pytest tests/ -v
```

### Train (mock models, quick demo)

```bash
python scripts/train.py --num-episodes 100
```

### Train (full curriculum with real models)

```bash
# Phase 0+1: Babbling + Vowels (~10-20 GPU-hours)
python scripts/run_babbling_vowels.py --use-real-sparc --use-real-sylber --device cuda

# Phase 2: Syllables (~40-60 GPU-hours)
python scripts/run_syllables.py --checkpoint checkpoints/checkpoint_ep6000.pt \
  --use-real-sparc --use-real-sylber --device cuda

# Ablation experiments
python scripts/run_ablations.py --num-episodes 5000 --device cuda
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_ep100.pt
```

### Generate Paper Figures

```bash
# From mock data (demo)
python scripts/generate_figures.py

# From real ablation results
python scripts/generate_figures.py --from-results results/ablations/ablation_results.json
```

## Architecture

```
Target Audio/Video
       |
       v
Observation Encoder --> RL Policy (PPO, MLP Actor-Critic, <1M params)
       |                      |
       |                 13-DOF Actions
       |                      |
       |                      v
       |               MuJoCo Oral Cavity (500 Hz physics)
       |                   /        \
       |            SPARC Decoder    Mesh Renderer
       |              (audio)         (visual)
       v                 \           /
  Sylber + LVE       Multi-Modal Reward
  Reward Module    (0.7*audio + 0.3*visual + aux)
       |
       v
  Policy Update (PPO)
```

## MDP Formulation

| Component | Specification |
|-----------|---------------|
| State | 39-dim: 13 positions + 13 velocities + 13 prev actions |
| Action | 13-dim, bounds [-0.5, 0.5] |
| Observation | 1353-dim: 15 frame-stacked states (585) + 768-dim Sylber target |
| Reward | R = 0.7 * cos_sim(Sylber) + 0.3 * (1 - LVE_norm) + R_aux |
| Episode | 50 steps = 2 seconds at 25 Hz, gamma = 0.99 |

### 13-DOF Articulators

| DOF | Articulator | Parameter |
|-----|------------|-----------|
| 1-2 | Tongue Dorsum | X, Y |
| 3-4 | Tongue Blade | X, Y |
| 5-6 | Tongue Tip | X, Y |
| 7-8 | Jaw | X, Y |
| 9-10 | Upper Lip | X, Y |
| 11-12 | Lower Lip | X, Y |
| 13 | Vocal Loudness | Scalar |

## Curriculum

| Phase | Episodes | Reward Mode | Goal |
|-------|----------|-------------|------|
| Babbling | 1,000 | Binary sound | Discover vocalization |
| Vowels | 5,000 | Audio-only | 5 cardinal vowels |
| Syllables | 25,000 | Combined | CV/CVC imitation |
| Person-specific | 25,000 | Combined | Speaker imitation |

## Project Structure

```
openjaw/
  core/          # MDP formulation, types, constants
  env/           # MuJoCo oral cavity environment
    assets/      # MuJoCo XML model
  audio/         # SPARC audio decoder integration
  visual/        # Mesh rendering, FLAME adapter
  perception/    # Frozen encoders (Sylber, wav2vec)
  reward/        # Multi-modal reward (audio, visual, auxiliary)
  policy/        # MLP Actor-Critic networks
  training/      # PPO trainer, curriculum, logger
  data/          # VOCASET + custom speaker loaders
  evaluation/    # Metrics, ablations, visualization

scripts/
  train.py                # General training entry point
  evaluate.py             # Evaluation entry point
  run_babbling_vowels.py  # Phase 0+1 training
  run_syllables.py        # Phase 2 training
  run_ablations.py        # Ablation experiments
  generate_figures.py     # Paper figure generation

paper/
  main.tex         # LaTeX paper
  references.bib   # Bibliography
  figures/         # Generated figures (PDF)

configs/
  default.yaml     # Full configuration

tests/             # 201 tests covering all modules
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| torch | Neural networks, PPO training |
| mujoco | Physics simulation |
| gymnasium | RL environment interface |
| librosa | Audio feature extraction (MCD) |
| matplotlib | Visualization |
| tensorboard | Training logging |

Optional (for real model inference):
| Package | Purpose |
|---------|---------|
| sparc (`speech-articulatory-coding`) | Real articulatory-to-audio decoding |
| sylber | Real syllabic embeddings |
| transformers | wav2vec 2.0, Whisper ASR |

## Ablation Conditions

| Condition | w_audio | w_visual | Curriculum |
|-----------|---------|----------|------------|
| Combined (default) | 0.7 | 0.3 | Yes |
| Audio-only | 1.0 | 0.0 | Yes |
| Visual-only | 0.0 | 1.0 | Yes |
| No curriculum | 0.7 | 0.3 | No |

## Reproducibility

- **Random seed:** 42 (default, configurable)
- **Hardware:** Single NVIDIA RTX 3090 or 4090 (24 GB VRAM)
- **Expected training time:** ~100 GPU-hours for full curriculum
- **Tests:** 201 unit + integration tests, all passing
- **Python:** 3.12+

All configuration is in `configs/default.yaml`. Override via command-line arguments.

## References

- Anand et al. (2025) - Articulatory RL with SPARC
- Todorov et al. (2012) - MuJoCo physics engine
- Schulman et al. (2017) - PPO algorithm
- Cho et al. (2024) - Sylber syllabic embeddings
- SPARC (2024) - Speech Articulatory Coding
- VOCASET / Cudeiro et al. (2019) - Audio-facial motion dataset
- FLAME / Li et al. (2017) - 3D morphable face model

See `paper/references.bib` for the complete bibliography.

## License

MIT License. See [LICENSE](LICENSE).
