import os
import unittest

from dataset import MusicBenchDataset


class DatasetPathTests(unittest.TestCase):
    def test_preserves_audio_subdirectories(self):
        for location, directory in (
            ("data/example.wav", "data"),
            ("data_aug2/example.wav", "data_aug2"),
            ("data_aug/example.wav", "data_aug"),
            ("./dataset/data_aug2/example.wav", "data_aug2"),
            (r"dataset\data\example.wav", "data"),
        ):
            with self.subTest(location=location):
                self.assertEqual(
                    MusicBenchDataset.normalize_audio_path("dataset", location),
                    os.path.join("dataset", directory, "example.wav"),
                )

    def test_filename_remains_relative_to_audio_root(self):
        self.assertEqual(
            MusicBenchDataset.normalize_audio_path("audio-root", "example.wav"),
            os.path.join("audio-root", "example.wav"),
        )


if __name__ == "__main__":
    unittest.main()
