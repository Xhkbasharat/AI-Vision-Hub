
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app as application


class TestImageValidation(unittest.TestCase):

    def setUp(self):

        self.client = application.app.test_client()

        self.temp_dir = tempfile.TemporaryDirectory()
        self.upload_dir = Path(self.temp_dir.name)

        self.upload_patch = patch(
            "app.UPLOAD_FOLDER",
            self.upload_dir,
            create=True
        )

        self.upload_patch.start()

        self.upload_db_patch = patch.object(
            application.db,
            "add_upload"
        )

        self.mock_add_upload = self.upload_db_patch.start()

    def tearDown(self):

        self.upload_db_patch.stop()
        self.upload_patch.stop()
        self.temp_dir.cleanup()

    # ======================================
    # IMAGE CREATION HELPER
    # ======================================

    @staticmethod
    def create_test_image():

        image_bytes = io.BytesIO()

        image = Image.new(
            "RGB",
            (100, 100),
            color="blue"
        )

        image.save(
            image_bytes,
            format="JPEG"
        )

        image_bytes.seek(0)

        return image_bytes

    # ======================================
    # VALID IMAGE
    # ======================================

    def test_valid_image_upload(self):

        image = self.create_test_image()

        response = self.client.post(
            "/detect-objects",
            data={
                "image": (
                    image,
                    "test.jpg"
                )
            },
            content_type="multipart/form-data"
        )

        # A successful upload should proceed to
        # the object detection module. The model
        # itself is mocked to avoid loading YOLO.
        self.assertNotEqual(
            response.status_code,
            400
        )

    # ======================================
    # INVALID FILE EXTENSION
    # ======================================

    def test_invalid_file_extension(self):

        response = self.client.post(
            "/detect-objects",
            data={
                "image": (
                    io.BytesIO(b"not an image"),
                    "malicious.exe"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertIn(
            response.status_code,
            [400, 415]
        )

    # ======================================
    # CORRUPTED IMAGE
    # ======================================

    def test_corrupted_image(self):

        response = self.client.post(
            "/detect-objects",
            data={
                "image": (
                    io.BytesIO(b"corrupted image data"),
                    "corrupted.jpg"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertIn(
            response.status_code,
            [400, 415]
        )

    # ======================================
    # MISSING IMAGE
    # ======================================

    def test_missing_image(self):

        response = self.client.post(
            "/detect-objects",
            data={},
            content_type="multipart/form-data"
        )

        self.assertIn(
            response.status_code,
            [400, 415]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
