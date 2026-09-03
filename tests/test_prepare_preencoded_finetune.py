import tempfile
import unittest
from pathlib import Path

from prepare_preencoded_finetune import (
    frame_count,
    resolve_raw_yuv,
    resolve_reconstructed_yuv,
    validate_disjoint_ids,
    yuv420p_frame_bytes,
)


class PreparePreencodedFinetuneTests(unittest.TestCase):
    def test_split_ids_must_be_video_disjoint(self):
        with self.assertRaisesRegex(ValueError, "train/test"):
            validate_disjoint_ids([1, 2], [3], [2, 4])

    def test_yuv_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1.yuv"
            bytes_per_frame = yuv420p_frame_bytes(4, 4)
            with path.open("wb") as yuv_file:
                yuv_file.truncate(bytes_per_frame * 5)
            self.assertEqual(frame_count(path, bytes_per_frame), 5)

    def test_resolves_unpadded_ldv_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            reconstructed = root / "reconstructed"
            raw.mkdir()
            reconstructed.mkdir()
            raw_path = raw / "7.yuv"
            reconstructed_path = reconstructed / "7_37rec.yuv"
            raw_path.touch()
            reconstructed_path.touch()
            self.assertEqual(resolve_raw_yuv(raw, 7), raw_path)
            self.assertEqual(
                resolve_reconstructed_yuv(reconstructed, 7, 37),
                reconstructed_path,
            )


if __name__ == "__main__":
    unittest.main()
