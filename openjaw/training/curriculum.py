"""Curriculum scheduler for progressive training difficulty.

Phases:
  0: Babbling     — random exploration, binary sound reward
  1: Vowels       — 5 cardinal vowels, audio-only reward
  2: Syllables    — CV/CVC syllables, combined audio+visual reward
  3: Person-specific — target speaker imitation, full reward
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CurriculumPhase:
    """Single curriculum phase definition."""

    name: str
    episodes: int
    reward_mode: str  # "binary_sound", "audio_only", "combined"
    targets: list[str] = field(default_factory=list)


# Default curriculum from PRD-C
DEFAULT_PHASES = [
    CurriculumPhase(
        name="babbling",
        episodes=1000,
        reward_mode="binary_sound",
    ),
    CurriculumPhase(
        name="vowels",
        episodes=5000,
        reward_mode="audio_only",
        targets=["a", "e", "i", "o", "u"],
    ),
    CurriculumPhase(
        name="syllables",
        episodes=25000,
        reward_mode="combined",
    ),
    CurriculumPhase(
        name="person_specific",
        episodes=25000,
        reward_mode="combined",
    ),
]


class CurriculumScheduler:
    """Manages curriculum phase transitions based on episode count."""

    def __init__(self, phases: list[CurriculumPhase] | None = None) -> None:
        self.phases = phases or DEFAULT_PHASES
        self._current_phase_idx = 0
        self._episodes_in_phase = 0
        self._total_episodes = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self._current_phase_idx]

    @property
    def phase_index(self) -> int:
        return self._current_phase_idx

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def is_complete(self) -> bool:
        return self._current_phase_idx >= len(self.phases)

    def step(self) -> bool:
        """Advance one episode. Returns True if phase changed."""
        if self.is_complete:
            return False

        self._episodes_in_phase += 1
        self._total_episodes += 1

        if self._episodes_in_phase >= self.current_phase.episodes:
            self._current_phase_idx += 1
            self._episodes_in_phase = 0
            return True

        return False

    def reset(self) -> None:
        """Reset to beginning of curriculum."""
        self._current_phase_idx = 0
        self._episodes_in_phase = 0
        self._total_episodes = 0

    def progress(self) -> dict[str, float | str | int]:
        """Get curriculum progress info."""
        if self.is_complete:
            return {
                "phase": "complete",
                "phase_idx": len(self.phases),
                "episodes_in_phase": 0,
                "total_episodes": self._total_episodes,
                "phase_progress": 1.0,
            }
        return {
            "phase": self.current_phase.name,
            "phase_idx": self._current_phase_idx,
            "episodes_in_phase": self._episodes_in_phase,
            "total_episodes": self._total_episodes,
            "phase_progress": self._episodes_in_phase / self.current_phase.episodes,
        }
