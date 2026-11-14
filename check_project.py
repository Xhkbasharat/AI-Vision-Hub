
from pathlib import Path
import sys


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "app.py",
    "config.py",
    "database.py",
    "requirements.txt",
    "train_fake_news.py",
    "test_database.py",
    "test_app.py",
    "test_image_validation.py",
    "test_config.py",
    "test_error_handler.py",
    "test_logger.py",
    "check_imports.py",
]

REQUIRED_FOLDERS = [
    "modules",
    "models",
    "templates",
    "static",
    "data",
    "data/uploads",
    "data/known_faces",
]

REQUIRED_MODULES = [
    "modules/__init__.py",
    "modules/face_recognition.py",
    "modules/object_detection.py",
    "modules/image_classification.py",
    "modules/ml_prediction.py",
    "modules/sentiment_analysis.py",
    "modules/fake_news.py",
    "modules/chatbot.py",
    "modules/error_handler.py",
    "modules/logger.py",
]

REQUIRED_FRONTEND_FILES = [
    "templates/index.html",
    "static/style.css",
]


# ==========================================
# CHECK FUNCTIONS
# ==========================================

def check_files():

    print("\nChecking required files...")

    missing = []

    for filename in REQUIRED_FILES:

        path = BASE_DIR / filename

        if path.is_file():

            print(f"[OK] {filename}")

        else:

            print(f"[MISSING] {filename}")

            missing.append(filename)

    return missing


def check_folders():

    print("\nChecking required folders...")

    missing = []

    for folder in REQUIRED_FOLDERS:

        path = BASE_DIR / folder

        if path.is_dir():

            print(f"[OK] {folder}/")

        else:

            print(f"[MISSING] {folder}/")

            missing.append(folder)

    return missing


def check_modules():

    print("\nChecking AI modules...")

    missing = []

    for module in REQUIRED_MODULES:

        path = BASE_DIR / module

        if path.is_file():

            print(f"[OK] {module}")

        else:

            print(f"[MISSING] {module}")

            missing.append(module)

    return missing


def check_frontend():

    print("\nChecking frontend files...")

    missing = []

    for filename in REQUIRED_FRONTEND_FILES:

        path = BASE_DIR / filename

        if path.is_file():

            print(f"[OK] {filename}")

        else:

            print(f"[MISSING] {filename}")

            missing.append(filename)

    return missing


# ==========================================
# MAIN VERIFICATION
# ==========================================

def main():

    print("=" * 55)
    print("          AI VISION HUB PROJECT CHECK")
    print("=" * 55)

    print(f"\nProject directory: {BASE_DIR}")

    missing_files = check_files()
    missing_folders = check_folders()
    missing_modules = check_modules()
    missing_frontend = check_frontend()

    all_missing = (
        missing_files
        + missing_folders
        + missing_modules
        + missing_frontend
    )

    print("\n" + "=" * 55)
    print("                 CHECK SUMMARY")
    print("=" * 55)

    if all_missing:

        print(
            f"\n[FAILED] {len(all_missing)} required items are missing."
        )

        print("\nMissing items:")

        for item in all_missing:

            print(f"- {item}")

        print(
            "\nCreate the missing files or folders before proceeding."
        )

        return 1

    print("\n[SUCCESS] All required project items exist.")

    print("\nAI Vision Hub project structure is complete.")

    return 0


if __name__ == "__main__":

    sys.exit(main())
