
import sys
import unittest
from pathlib import Path


# ==========================================
# TEST RUNNER CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

TEST_FILES = [
    "test_config.py",
    "test_database.py",
    "test_error_handler.py",
    "test_logger.py",
    "test_image_validation.py",
    "test_app.py",
]



# ==========================================
# TEST DISCOVERY
# ==========================================

def discover_tests():

    loader = unittest.TestLoader()

    suite = unittest.TestSuite()

    missing_files = []

    print("\nChecking test files...\n")

    for filename in TEST_FILES:

        test_path = BASE_DIR / filename

        if not test_path.is_file():

            print(f"[MISSING] {filename}")

            missing_files.append(filename)

            continue

        print(f"[FOUND] {filename}")

        module_name = filename[:-3]

        try:

            discovered_suite = loader.discover(
                start_dir=str(BASE_DIR),
                pattern=filename,
                top_level_dir=str(BASE_DIR),
            )

            suite.addTests(discovered_suite)

        except Exception as error:

            print(
                f"[ERROR] Unable to discover {module_name}: {error}"
            )

    return suite, missing_files


# ==========================================
# RUN TESTS
# ==========================================

def run_all_tests():

    print("=" * 60)
    print("             AI VISION HUB TEST RUNNER")
    print("=" * 60)

    suite, missing_files = discover_tests()

    if missing_files:

        print("\nWARNING: Some test files are missing.")

        for filename in missing_files:

            print(f"- {filename}")

    total_tests = suite.countTestCases()

    if total_tests == 0:

        print("\nNo tests were discovered.")

        return 1

    print("\n" + "=" * 60)
    print(f"Running {total_tests} tests...")
    print("=" * 60)

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(suite)

    # ======================================
    # SUMMARY
    # ======================================

    print("\n" + "=" * 60)
    print("                 TEST SUMMARY")
    print("=" * 60)

    print(f"Total tests : {result.testsRun}")
    print(f"Passed      : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed      : {len(result.failures)}")
    print(f"Errors      : {len(result.errors)}")
    print(f"Skipped     : {len(result.skipped)}")

    if missing_files:

        print(f"Missing test files: {len(missing_files)}")

    if result.wasSuccessful() and not missing_files:

        print("\n[SUCCESS] All discovered tests passed.")

        return 0

    print("\n[FAILED] Some tests failed or files are missing.")

    return 1


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    sys.exit(
        run_all_tests()
    )
