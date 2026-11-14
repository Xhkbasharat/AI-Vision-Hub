
import io
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import app, save_image, database


class TestImageValidation(unittest.TestCase):

    def setUp(self):

        self.temp_directory = tempfile.TemporaryDirectory()

        self.upload_directory = Path(
            self.temp_directory.name
        )

        self.app_patcher = patch(
            "app.UPLOAD_FOLDER",
            self.upload_directory
        )

        self.app_patcher.start()

        app.config["TESTING"] = True

        self.client = app.test_client()

    def tearDown(self):

        self.app_patcher.stop()

        self.temp_directory.cleanup()

    # ======================================
    # CREATE TEST IMAGE
    # ======================================

    @staticmethod
    def create_image():

        image = Image.new(
            "RGB",
            (100, 100),
            color="blue"
        )

        image_stream = io.BytesIO()

        image.save(
            image_stream,
            format="JPEG"
        )

        image_stream.seek(0)

        return image_stream

    # ======================================
    # VALID IMAGE TEST
    # ======================================

    @patch.object(database, "add_upload")
    def test_valid_image(self, mock_add_upload):

        image_stream = self.create_image()

        from werkzeug.datastructures import FileStorage

        uploaded_file = FileStorage(
            stream=image_stream,
            filename="test.jpg",
            content_type="image/jpeg"
        )

        saved_path = save_image(
            uploaded_file
        )

        self.assertTrue(
            Path(saved_path).exists()
        )

        self.assertEqual(
            Path(saved_path).suffix,
            ".jpg"
        )

        mock_add_upload.assert_called_once()

    # ======================================
    # MISSING IMAGE TEST
    # ======================================

    def test_missing_image(self):

        with self.assertRaises(ValueError):

            save_image(None)

    # ======================================
    # EMPTY FILENAME TEST
    # ======================================

    def test_empty_filename(self):

        from werkzeug.datastructures import FileStorage

        uploaded_file = FileStorage(
            stream=io.BytesIO(b""),
            filename=""
        )

        with self.assertRaises(ValueError):

            save_image(uploaded_file)

    # ======================================
    # INVALID EXTENSION TEST
    # ======================================

    def test_invalid_extension(self):

        from werkzeug.datastructures import FileStorage

        uploaded_file = FileStorage(
            stream=io.BytesIO(b"invalid"),
            filename="document.txt"
        )

        with self.assertRaises(ValueError):

            save_image(uploaded_file)

    # ======================================
    # CORRUPTED IMAGE TEST
    # ======================================

    def test_corrupted_image(self):

        from werkzeug.datastructures import FileStorage

        uploaded_file = FileStorage(
            stream=io.BytesIO(
                b"this is not a valid image"
            ),
            filename="corrupted.jpg"
        )

        with self.assertRaises(ValueError):

            save_image(uploaded_file)

    # ======================================
    # IMAGE CONTENT MISMATCH TEST
    # ======================================

    def test_extension_content_mismatch(self):

        from werkzeug.datastructures import FileStorage

        # Valid PNG data with a JPG extension.
        image = Image.new(
            "RGB",
            (50, 50),
            color="red"
        )

        stream = io.BytesIO()

        image.save(
            stream,
            format="PNG"
        )

        stream.seek(0)

        uploaded_file = FileStorage(
            stream=stream,
            filename="fake.jpg",
            content_type="image/jpeg"
        )

        # The current implementation verifies image content
        # but does not require the extension to match it.
        saved_path = save_image(
            uploaded_file
        )

        self.assertTrue(
            Path(saved_path).exists()
        )

    # ======================================
    # IMAGE UPLOAD ROUTE TEST
    # ======================================

    def test_upload_route_missing_image(self):

        response = self.client.post(
            "/detect-objects",
            data={}
        )

        self.assertEqual(
            response.status_code,
            400
        )

    # ======================================
    # IMAGE UPLOAD ROUTE INVALID FILE
    # ======================================

    def test_upload_route_invalid_file(self):

        response = self.client.post(
            "/classify-image",
            data={
                "image": (
                    io.BytesIO(b"invalid image"),
                    "document.txt"
                )
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(
            response.status_code,
            400
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
