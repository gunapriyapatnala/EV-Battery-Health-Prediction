import joblib
import glob
import os

for file in glob.glob("models/*.joblib"):
    print("\nTESTING:", os.path.basename(file))
    try:
        joblib.load(file)
        print("OK")
    except Exception as e:
        print("FAILED:", type(e).__name__)
        print("ERROR:", e)