
import importlib
import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger("AI-Vision-Hub")


# ======================================
# MODULES TO CHECK
# ======================================

MODULES = [
    "config",
    "database",
    "app",
    "modules.face_recognition",
    "modules.object_detection",
    "modules.image_classification",
    "modules.ml_prediction",
    "modules.sentiment_analysis",
    "modules.fake_news",
    "modules.chatbot",
    "modules.error_handler",
    "modules.logger"
]


# ======================================
# EXPECTED CLASSES AND FUNCTIONS
# ======================================

EXPECTED_EXPORTS = {

    "database": [
        "Database"
    ],

    "modules.face_recognition": [
        "FaceRecognition"
    ],

    "modules.object_detection": [
        "ObjectDetection"
    ],

    "modules.image_classification": [
        "ImageClassification"
    ],

    "modules.ml_prediction": [
        "MLPrediction"
    ],

    "modules.sentiment_analysis": [
        "SentimentAnalyzer"
    ],

    "modules.fake_news": [
        "FakeNewsDetector"
    ],

    "modules.chatbot": [
        "Chatbot"
    ],

    "modules.error_handler": [
        "AIVisionHubError",
        "ImageValidationError",
        "ModelLoadingError",
        "PredictionError",
        "DatabaseError"
    ],

    "modules.logger": [
        "app_logger",
        "log_info",
        "log_warning",
        "log_error"
    ]
}


# ======================================
# CHECK IMPORTS
# ======================================

def check_imports():

    passed = 0
    failed = 0

    print("\n===================================")
    print(" AI VISION HUB - IMPORT CHECK")
    print("===================================\n")

    for module_name in MODULES:

        try:

            module = importlib.import_module(
                module_name
            )

            missing_exports = []

            expected = EXPECTED_EXPORTS.get(
                module_name,
                []
            )

            for export in expected:

                if not hasattr(module, export):

                    missing_exports.append(
                        export
                    )

            if missing_exports:

                failed += 1

                print(
                    f"[FAIL] {module_name}"
                )

                print(
                    "Missing exports:",
                    ", ".join(missing_exports)
                )

                continue

            passed += 1

            print(
                f"[PASS] {module_name}"
            )

        except Exception as error:

            failed += 1

            print(
                f"[FAIL] {module_name}"
            )

            print(
                f"       {type(error).__name__}: {error}"
            )

            logger.exception(
                "Import failed: %s",
                module_name
            )

    # ==================================
    # SUMMARY
    # ==================================

    total = passed + failed

    print("\n===================================")
    print(" IMPORT CHECK SUMMARY")
    print("===================================")

    print(
        f"Total modules: {total}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print("===================================\n")

    if failed:

        print(
            "Import verification failed. "
            "Review the errors above."
        )

        return False

    print(
        "All module imports verified successfully."
    )

    return True


# ======================================
# MAIN
# ======================================

if __name__ == "__main__":

    success = check_imports()

    sys.exit(
        0 if success else 1
    )
