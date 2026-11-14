
import importlib
import inspect


MODULES = {
    "modules.face_recognition": "FaceRecognition",
    "modules.object_detection": "ObjectDetection",
    "modules.image_classification": "ImageClassification",
    "modules.ml_prediction": "MLPrediction",
    "modules.sentiment_analysis": "SentimentAnalyzer",
    "modules.fake_news": "FakeNewsDetector",
    "modules.chatbot": "Chatbot",
}


def check_imports():

    print("\n================================")
    print(" AI VISION HUB IMPORT CHECK")
    print("================================\n")

    passed = 0
    failed = 0

    for module_name, class_name in MODULES.items():

        try:

            module = importlib.import_module(module_name)

            model_class = getattr(
                module,
                class_name,
                None
            )

            if model_class is None:

                print(
                    f"[FAILED] {module_name}: "
                    f"{class_name} not found"
                )

                failed += 1
                continue

            if not inspect.isclass(model_class):

                print(
                    f"[FAILED] {class_name} is not a class"
                )

                failed += 1
                continue

            print(
                f"[OK] {module_name}.{class_name}"
            )

            passed += 1

        except Exception as error:

            print(
                f"[FAILED] {module_name}: {error}"
            )

            failed += 1

    print("\n================================")
    print(" IMPORT CHECK RESULTS")
    print("================================")

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:

        print("\nAll expected module imports passed.")

    else:

        print("\nFix the failed imports before starting Flask.")

    print("================================\n")


if __name__ == "__main__":
    check_imports()
