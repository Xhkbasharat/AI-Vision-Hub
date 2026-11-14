
import os
import subprocess
import sys
import unittest

from pathlib import Path

import config


class TestAIVisionHubConfig(unittest.TestCase):

    # ======================================
    # DIRECTORY TESTS
    # ======================================

    def test_base_directory_exists(self):

        self.assertTrue(config.BASE_DIR.exists())

    def test_data_directory_exists(self):

        self.assertTrue(config.DATA_DIR.exists())

    def test_upload_directory_exists(self):

        self.assertTrue(config.UPLOAD_FOLDER.exists())

    def test_known_faces_directory_exists(self):

        self.assertTrue(config.KNOWN_FACES_FOLDER.exists())

    def test_model_directory_exists(self):

        self.assertTrue(config.MODEL_FOLDER.exists())

    # ======================================
    # APPLICATION SETTINGS
    # ======================================

    def test_application_name(self):

        self.assertEqual(
            config.APP_NAME,
            "AI Vision Hub"
        )

    def test_application_version(self):

        self.assertEqual(
            config.APP_VERSION,
            "1.0.0"
        )

    def test_upload_size_limit(self):

        self.assertGreater(
            config.MAX_CONTENT_LENGTH,
            0
        )

    def test_allowed_image_extensions(self):

        expected = {
            "jpg",
            "jpeg",
            "png",
            "webp"
        }

        self.assertEqual(
            config.ALLOWED_IMAGE_EXTENSIONS,
            expected
        )

    # ======================================
    # SECRET KEY VALIDATION
    # ======================================

    def test_secret_key_length(self):

        self.assertGreaterEqual(
            len(config.SECRET_KEY),
            32
        )

    # ======================================
    # PRODUCTION SECURITY TESTS
    # ======================================

    def run_config_subprocess(self, environment):

        env = os.environ.copy()

        env.update(environment)

        return subprocess.run(
            [
                sys.executable,
                "-c",
                "import config"
            ],
            capture_output=True,
            text=True,
            env=env
        )

    def test_production_requires_secret_key(self):

        result = self.run_config_subprocess({
            "APP_ENV": "production",
            "SECRET_KEY": "",
            "FLASK_DEBUG": "false"
        })

        self.assertNotEqual(
            result.returncode,
            0
        )

        self.assertIn(
            "SECRET_KEY",
            result.stderr
        )

    def test_production_rejects_debug_mode(self):

        result = self.run_config_subprocess({
            "APP_ENV": "production",
            "SECRET_KEY": "a" * 64,
            "FLASK_DEBUG": "true"
        })

        self.assertNotEqual(
            result.returncode,
            0
        )

        self.assertIn(
            "Debug mode",
            result.stderr
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
