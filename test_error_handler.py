
import unittest

from modules.error_handler import (
    AIVisionHubError,
    ImageValidationError,
    ModelLoadingError,
    PredictionError,
    DatabaseError,
)


class TestErrorHandler(unittest.TestCase):

    # ======================================
    # BASE APPLICATION ERROR
    # ======================================

    def test_base_error(self):

        error = AIVisionHubError(
            "Application error"
        )

        self.assertIsInstance(
            error,
            Exception
        )

        self.assertEqual(
            str(error),
            "Application error"
        )

    # ======================================
    # IMAGE VALIDATION ERROR
    # ======================================

    def test_image_validation_error(self):

        error = ImageValidationError(
            "Invalid image"
        )

        self.assertIsInstance(
            error,
            AIVisionHubError
        )

        self.assertEqual(
            str(error),
            "Invalid image"
        )

    # ======================================
    # MODEL LOADING ERROR
    # ======================================

    def test_model_loading_error(self):

        error = ModelLoadingError(
            "Unable to load model"
        )

        self.assertIsInstance(
            error,
            AIVisionHubError
        )

        self.assertEqual(
            str(error),
            "Unable to load model"
        )

    # ======================================
    # PREDICTION ERROR
    # ======================================

    def test_prediction_error(self):

        error = PredictionError(
            "Prediction failed"
        )

        self.assertIsInstance(
            error,
            AIVisionHubError
        )

        self.assertEqual(
            str(error),
            "Prediction failed"
        )

    # ======================================
    # DATABASE ERROR
    # ======================================

    def test_database_error(self):

        error = DatabaseError(
            "Database operation failed"
        )

        self.assertIsInstance(
            error,
            AIVisionHubError
        )

        self.assertEqual(
            str(error),
            "Database operation failed"
        )

    # ======================================
    # EXCEPTION RAISING
    # ======================================

    def test_raise_image_validation_error(self):

        with self.assertRaises(
            ImageValidationError
        ):

            raise ImageValidationError(
                "Unsupported image format"
            )

    def test_raise_model_loading_error(self):

        with self.assertRaises(
            ModelLoadingError
        ):

            raise ModelLoadingError(
                "Model file not found"
            )

    def test_raise_prediction_error(self):

        with self.assertRaises(
            PredictionError
        ):

            raise PredictionError(
                "Unable to generate prediction"
            )

    def test_raise_database_error(self):

        with self.assertRaises(
            DatabaseError
        ):

            raise DatabaseError(
                "Unable to save database record"
            )

    # ======================================
    # ERROR INHERITANCE
    # ======================================

    def test_all_custom_errors_inherit_base(self):

        error_classes = [
            ImageValidationError,
            ModelLoadingError,
            PredictionError,
            DatabaseError,
        ]

        for error_class in error_classes:

            with self.subTest(
                error_class=error_class.__name__
            ):

                self.assertTrue(
                    issubclass(
                        error_class,
                        AIVisionHubError
                    )
                )

    # ======================================
    # ERROR MESSAGE PRESERVATION
    # ======================================

    def test_error_message_preservation(self):

        message = "A detailed application error"

        error = PredictionError(message)

        self.assertEqual(
            error.args[0],
            message
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
