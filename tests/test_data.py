"""Step 11: Data pipeline tests (preprocessing, vocaset, custom_speaker)."""

from __future__ import annotations

import numpy as np
import pytest

from openjaw.core.types import AUDIO_SAMPLE_RATE
from openjaw.data.custom_speaker import CustomSpeakerLoader
from openjaw.data.preprocessing import SyllableTarget, create_vowel_targets, segment_audio_to_syllables
from openjaw.data.vocaset import VOCASETLoader
from openjaw.perception.sylber import MockSylberEncoder


# ── Preprocessing ───────────────────────────────────────────────────────────

class TestSyllableTarget:
    def test_dataclass_fields(self):
        emb = np.zeros(768, dtype=np.float32)
        audio = np.zeros(8000, dtype=np.float32)
        target = SyllableTarget(
            label="ba", audio_embedding=emb, audio_segment=audio,
            lip_vertices=None, start_time=0.0, end_time=0.5,
        )
        assert target.label == "ba"
        assert target.audio_embedding.shape == (768,)
        assert target.lip_vertices is None


class TestSegmentAudio:
    def test_segment_returns_list(self):
        encoder = MockSylberEncoder()
        # Generate audio with some energy (sine wave)
        t = np.arange(16000) / 16000
        audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        targets = segment_audio_to_syllables(audio, encoder)
        assert isinstance(targets, list)
        for tgt in targets:
            assert isinstance(tgt, SyllableTarget)
            assert tgt.audio_embedding.shape == (768,)
            assert len(tgt.audio_segment) > 0
            assert tgt.end_time > tgt.start_time

    def test_silent_audio_may_yield_empty(self):
        encoder = MockSylberEncoder()
        audio = np.zeros(8000, dtype=np.float32)
        targets = segment_audio_to_syllables(audio, encoder)
        # Silent audio might still produce segments from mock encoder
        assert isinstance(targets, list)

    def test_segment_embedding_shape(self):
        encoder = MockSylberEncoder()
        t = np.arange(32000) / 16000
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        targets = segment_audio_to_syllables(audio, encoder)
        for tgt in targets:
            assert tgt.audio_embedding.dtype == np.float32
            assert tgt.audio_embedding.shape == (768,)


class TestCreateVowelTargets:
    def test_default_five_vowels(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder)
        assert len(targets) == 5
        labels = {t.label for t in targets}
        assert labels == {"a", "e", "i", "o", "u"}

    def test_custom_vowels(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder, vowels=["a", "o"])
        assert len(targets) == 2

    def test_embedding_shape(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder)
        for t in targets:
            assert t.audio_embedding.shape == (768,)
            assert t.audio_embedding.dtype == np.float32

    def test_audio_segment_not_empty(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder)
        for t in targets:
            assert len(t.audio_segment) > 0
            assert t.audio_segment.dtype == np.float32

    def test_different_vowels_different_embeddings(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder)
        embs = [t.audio_embedding for t in targets]
        # At least some embeddings should differ (different audio → different mock output)
        different = False
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                if not np.allclose(embs[i], embs[j]):
                    different = True
                    break
        assert different, "All vowel embeddings are identical"

    def test_time_bounds(self):
        encoder = MockSylberEncoder()
        targets = create_vowel_targets(encoder, duration=0.5)
        for t in targets:
            assert t.start_time == 0.0
            assert t.end_time == 0.5


# ── VOCASET ─────────────────────────────────────────────────────────────────

class TestVOCASETLoader:
    def test_unavailable_when_no_dir(self, tmp_path):
        loader = VOCASETLoader(data_dir=str(tmp_path / "nonexistent"))
        assert not loader.is_available
        assert loader.num_sequences == 0

    def test_available_with_dir(self, tmp_path):
        data_dir = tmp_path / "vocaset"
        (data_dir / "audio").mkdir(parents=True)
        loader = VOCASETLoader(data_dir=str(data_dir))
        assert loader.is_available
        assert loader.num_sequences == 0  # No WAV files yet

    def test_create_mock_sequence(self):
        seq = VOCASETLoader.create_mock_sequence(duration=1.0)
        assert seq["name"] == "mock_sequence"
        assert seq["audio"].dtype == np.float32
        assert len(seq["audio"]) == 16000  # 1.0s * 16kHz
        assert seq["vertices"] is not None
        assert seq["vertices"].shape[0] == 60  # 1.0s * 60 FPS
        assert seq["vertices"].shape[2] == 3

    def test_mock_sequence_custom_params(self):
        seq = VOCASETLoader.create_mock_sequence(
            duration=2.0, sample_rate=8000, num_vertices=5,
        )
        assert len(seq["audio"]) == 16000  # 2.0s * 8kHz
        assert seq["vertices"].shape[1] == 5

    def test_load_invalid_index_raises(self, tmp_path):
        loader = VOCASETLoader(data_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(IndexError):
            loader.load_sequence(0)


# ── Custom Speaker ──────────────────────────────────────────────────────────

class TestCustomSpeakerLoader:
    def test_unavailable_when_no_dir(self, tmp_path):
        loader = CustomSpeakerLoader(data_dir=str(tmp_path / "nonexistent"))
        assert not loader.is_available
        assert loader.list_recordings() == []

    def test_available_with_dir(self, tmp_path):
        data_dir = tmp_path / "custom"
        data_dir.mkdir()
        loader = CustomSpeakerLoader(data_dir=str(data_dir))
        assert loader.is_available

    def test_create_mock_recording(self):
        audio, sr = CustomSpeakerLoader.create_mock_recording(duration=1.0)
        assert sr == AUDIO_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert len(audio) == AUDIO_SAMPLE_RATE  # 1.0s
        # Should contain non-zero samples (sine waves)
        assert np.max(np.abs(audio)) > 0.1

    def test_mock_recording_custom_duration(self):
        audio, sr = CustomSpeakerLoader.create_mock_recording(duration=0.5)
        assert len(audio) == int(0.5 * sr)
