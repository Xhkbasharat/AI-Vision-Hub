
import logging
import unittest
from unittest.mock import patch

from modules.logger import (
    app_logger,
    log_info,
    log_warning,
    log_error,
)


class TestLogger(unittest.TestCase):

    def test_logger_exists(self):

        self.assertIsInstance(
            app_logger,
            logging.Logger
        )

    def test_logger_has_name(self):

        self.assertTrue(
            app_logger.name
        )

    def test_log_info(self):

        with self.assertLogs(
            app_logger,
            level="INFO"
        ) as captured:

            log_info(
                "Test information message"
            )

        self.assertTrue(
            any(
                "Test information message" in message
                for message in captured.output
            )
        )

    def test_log_warning(self):

        with self.assertLogs(
            app_logger,
            level="WARNING"
        ) as captured:

            log_warning(
                "Test warning message"
            )

        self.assertTrue(
            any(
                "Test warning message" in message
                for message in captured.output
            )
        )

    def test_log_error(self):

        with self.assertLogs(
            app_logger,
            level="ERROR"
        ) as captured:

            log_error(
                "Test error message"
            )

        self.assertTrue(
            any(
                "Test error message" in message
                for message in captured.output
            )
        )

    def test_log_info_preserves_message(self):

        message = "Application started"

        with self.assertLogs(
            app_logger,
            level="INFO"
        ) as captured:

            log_info(message)

        self.assertIn(
            message,
            captured.output[0]
        )

    def test_log_warning_preserves_message(self):

        message = "Low disk space"

        with self.assertLogs(
            app_logger,
            level="WARNING"
        ) as captured:

            log_warning(message)

        self.assertIn(
            message,
            captured.output[0]
        )

    def test_log_error_preserves_message(self):

        message = "Database connection failed"

        with self.assertLogs(
            app_logger,
            level="ERROR"
        ) as captured:

            log_error(message)

        self.assertIn(
            message,
            captured.output[0]
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
