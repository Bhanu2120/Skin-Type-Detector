# Save this code in a new file named test_detector.py

import os
import cv2
import numpy as np
import requests

print("--- Starting Face Detector Test ---")

try:
    basedir = os.path.abspath(os.path.dirname(__file__))

    # --- 1. Load the Caffe DNN Face Detector Model ---
    prototxt_path = os.path.join(basedir, 'deploy.prototxt.txt')
    weights_path = os.path.join(basedir, 'res10_300x300_ssd_iter_140000.caffemodel')

    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        raise FileNotFoundError("ERROR: Model files not found. Make sure 'deploy.prototxt' and the '.caffemodel' file are in the same folder as this script.")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)
    print("✅ DNN model files loaded successfully.")

    # --- 2. Download a Test Image ---
    test_image_url = "https://www.biometricupdate.com/wp-content/uploads/2022/08/face-biometrics-on-a-woman-at-a-un-refugee-camp-scaled.jpg"
    response = requests.get(test_image_url)
    response.raise_for_status()
    print("✅ Test image downloaded successfully.")

    # --- 3. Run Face Detection ---
    filestr = response.content
    npimg = np.frombuffer(filestr, np.uint8)
    image_cv = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    (h, w) = image_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    highest_confidence = np.max(detections[0, 0, :, 2])

    # --- 4. Report the Result ---
    if highest_confidence > 0.5:
        print("\n==============================================")
        print(f"✅ TEST PASSED: Face detected with {highest_confidence:.2f} confidence.")
        print("This means your model files and OpenCV are working correctly.")
        print("==============================================")
    else:
        print("\n========================================================")
        print(f"❌ TEST FAILED: Face was NOT detected. Highest confidence was only {highest_confidence:.2f}.")
        print("This proves the problem is with your '.caffemodel' file.")
        print("It is likely corrupted or incomplete. Please re-download it.")
        print("========================================================")

except Exception as e:
    print(f"\nAN ERROR OCCURRED: {e}")