from pathlib import Path

from src.deployment.demo import CLASS_NAMES, SAMPLE_DIR, iter_demo_wavs


def test_iter_demo_wavs_reads_class_folders(tmp_path):
    (tmp_path / "dog_bark").mkdir()
    (tmp_path / "dog_bark" / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "siren").mkdir()
    (tmp_path / "siren" / "b.wav").write_bytes(b"RIFF")
    (tmp_path / "orphan.wav").write_bytes(b"RIFF")
    pairs = iter_demo_wavs(tmp_path)
    assert [(cls, path.name) for cls, path in pairs] == [
        ("dog_bark", "a.wav"),
        ("siren", "b.wav"),
    ]


def test_committed_sample_audio_has_three_per_class():
    pairs = iter_demo_wavs(SAMPLE_DIR)
    by_class: dict[str, int] = {name: 0 for name in CLASS_NAMES}
    for class_name, path in pairs:
        assert path.is_file()
        by_class[class_name] += 1
    assert by_class == {name: 3 for name in CLASS_NAMES}
    assert (SAMPLE_DIR / "dog_bark_sample.wav").is_file()
    assert Path(__file__).resolve().parents[1] / "data" / "sample_audio" == SAMPLE_DIR
