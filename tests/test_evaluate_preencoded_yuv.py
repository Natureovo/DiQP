import tempfile
import unittest
from pathlib import Path

from evaluate_preencoded_yuv import (
    aligned_frame_count,
    discover_pairs,
    resolve_raw_yuv,
    resolve_reconstructed_yuv,
    yuv420p_frame_bytes,
)


class PreencodedYuvTests(unittest.TestCase):
    def test_known_ldv_geometry_and_frame_count(self):
        frame_bytes = yuv420p_frame_bytes(960, 536)
        self.assertEqual(frame_bytes, 771840)
        self.assertEqual(frame_bytes * 120, 92620800)

    def test_pair_discovery_reports_unmatched_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            encoded = root / "encoded"
            raw.mkdir()
            encoded.mkdir()
            for video_id in (1, 2, 3):
                (raw / f"{video_id}.yuv").touch()
            for video_id in (1, 2, 4):
                (encoded / f"{video_id}_37rec.yuv").touch()
            (encoded / "1_37str.bin").touch()

            matched, raw_only, reconstructed_only = discover_pairs(raw, encoded, 37)

            self.assertEqual(matched, [1, 2])
            self.assertEqual(raw_only, [3])
            self.assertEqual(reconstructed_only, [4])

    def test_zero_padded_names_can_be_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            encoded = root / "encoded"
            raw.mkdir()
            encoded.mkdir()
            raw_path = raw / "001.yuv"
            reconstructed_path = encoded / "001_37rec.yuv"
            raw_path.touch()
            reconstructed_path.touch()

            self.assertEqual(resolve_raw_yuv(raw, 1), raw_path)
            self.assertEqual(
                resolve_reconstructed_yuv(encoded, 1, 37), reconstructed_path
            )

    def test_aligned_frame_count_checks_both_yuv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "100.yuv"
            reconstructed = root / "100_37rec.yuv"
            size = yuv420p_frame_bytes(4, 4) * 3
            with open(raw, "wb") as raw_file:
                raw_file.truncate(size)
            with open(reconstructed, "wb") as reconstructed_file:
                reconstructed_file.truncate(size)

            self.assertEqual(aligned_frame_count(raw, reconstructed, 4, 4), 3)

    def test_aligned_frame_count_uses_common_leading_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "20.yuv"
            reconstructed = root / "20_37rec.yuv"
            size = yuv420p_frame_bytes(4, 4)
            with open(raw, "wb") as raw_file:
                raw_file.truncate(size * 209)
            with open(reconstructed, "wb") as reconstructed_file:
                reconstructed_file.truncate(size * 208)

            self.assertEqual(aligned_frame_count(raw, reconstructed, 4, 4), 208)


if __name__ == "__main__":
    unittest.main()
