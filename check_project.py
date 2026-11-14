
import ast
from pathlib import Path


# ==========================================
# PROJECT FILES
# ==========================================

REQUIRED_FILES = [
    "app.py",
    "config.py",
    "database.py",
    "requirements.txt",
    "train_fake_news.py",
    "modules/__init__.py",
    "modules/face_recognition.py",
    "modules/object_detection.py",
    "modules/image_classification.py",
    "modules/ml_prediction.py",
    "modules/sentiment_analysis.py",
    "modules/fake_news.py",
    "modules/chatbot.py",
    "templates/index.html",
    "static/style.css",
    "README.md"
]


# ==========================================
# REQUIRED FUNCTIONS AND METHODS
# ==========================================

REQUIRED_METHODS = {
    "database.py": [
        "get_connection",
        "create_tables",
        "add_upload",
        "add_face",
        "get_faces",
        "save_analysis",
        "get_analysis_history",
        "get_dashboard_stats"
    ],

    "modules/face_recognition.py": [
        "register_face",
        "train_model",
        "recognize_face"
    ],

    "modules/object_detection.py": [
        "detect_objects"
    ],

    "modules/image_classification.py": [
        "classify_image",
        "get_best_prediction"
    ],

    "modules/ml_prediction.py": [
        "predict"
    ],

    "modules/sentiment_analysis.py": [
        "analyze_sentiment",
        "analyze_multiple"
    ],

    "modules/fake_news.py": [
        "predict",
        "load_model"
    ],

    "modules/chatbot.py": [
        "chat",
        "reset_chat"
    ],

    "train_fake_news.py": [
        "clean_text",
        "load_dataset",
        "train_model"
    ]
}


# ==========================================
# PYTHON SYNTAX CHECK
# ==========================================

def check_python_syntax(file_path):

    try:
        source = file_path.read_text(
            encoding="utf-8"
        )

        ast.parse(source)

        return True, "Syntax OK"

    except SyntaxError as error:

        return False, (
            f"Line {error.lineno}: {error.msg}"
        )

    except OSError as error:

        return False, str(error)


# ==========================================
# CLASS AND METHOD CHECK
# ==========================================

def check_required_methods(file_path, required_methods):

    source = file_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    available_methods = set()

    for node in ast.walk(tree):

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            available_methods.add(node.name)

    missing = [
        method
        for method in required_methods
        if method not in available_methods
    ]

    return missing


# ==========================================
# RUN PROJECT CHECKS
# ==========================================

def main():

    root = Path(__file__).resolve().parent

    passed = 0
    failed = 0

    print("\n================================")
    print("       AI VISION HUB CHECK")
    print("================================\n")

    # Check required files.
    print("1. FILE CHECK")

    for filename in REQUIRED_FILES:

        file_path = root / filename

        if file_path.exists():

            print(f"[OK] {filename}")
            passed += 1

        else:

            print(f"[MISSING] {filename}")
            failed += 1

    # Check Python syntax.
    print("\n2. PYTHON SYNTAX CHECK")

    python_files = [
        root / "app.py",
        root / "config.py",
        root / "database.py",
        root / "train_fake_news.py"
    ]

    python_files.extend(
        root / filename
        for filename in REQUIRED_METHODS
        if filename.startswith("modules/")
    )

    for file_path in python_files:

        if not file_path.exists():
            continue

        valid, message = check_python_syntax(file_path)

        if valid:

            print(f"[OK] {file_path.relative_to(root)}")
            passed += 1

        else:

            print(
                f"[ERROR] {file_path.relative_to(root)}: {message}"
            )
            failed += 1

    # Check required methods.
    print("\n3. METHOD CHECK")

    for filename, methods in REQUIRED_METHODS.items():

        file_path = root / filename

        if not file_path.exists():
            continue

        try:
            missing = check_required_methods(
                file_path,
                methods
            )

            if missing:

                print(
                    f"[ERROR] {filename}: Missing {missing}"
                )
                failed += len(missing)

            else:

                print(f"[OK] {filename}")
                passed += 1

        except SyntaxError:

            print(f"[SKIP] {filename}: Syntax error")
            failed += 1

    # Final result.
    print("\n================================")
    print("          FINAL REPORT")
    print("================================")

    print(f"Passed checks: {passed}")
    print(f"Failed checks: {failed}")

    if failed == 0:
        print("\nProject structure checks passed.")

    else:
        print("\nFix the reported issues before testing.")

    print("================================\n")


if __name__ == "__main__":
    main()
