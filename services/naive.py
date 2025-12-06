import numpy as np
import cv2 as cv
from flask.json import jsonify
from services.startup import clf, pca
from services.patch2vector import patch_to_vector
from services.detection_face import detect_face_and_all_features



def predict(file_storage):
    # Read file content to numpy array
    filestr = file_storage.read()
    npimg = np.frombuffer(filestr, np.uint8)
    # Decode image
    img_decoded = cv.imdecode(npimg, cv.IMREAD_COLOR)
    
    if img_decoded is None:
        raise ValueError("Could not decode image")

    detection = detect_face_and_all_features(img_decoded)
    
    if detection is None:
        return {"error": "No face detected or missing features"}
    

    feature_key = ["left_eye", "right_eye", "nose", "mouth"]
    test_vector_new = []
    for key in feature_key:
        vec_new = patch_to_vector(detection[key])
        projected_new = pca[key].transform([vec_new])[0]
        test_vector_new.extend(list(projected_new))

    test_vector_new = np.array(test_vector_new).reshape(1, -1)
    

    pred_gender_new = clf.predict(test_vector_new)[0]
    prob_new = clf.predict_proba(test_vector_new)[0]

    class_index_new = list(clf.classes_).index(pred_gender_new)
    score_new = prob_new[class_index_new]
    
    return {
        "gender": pred_gender_new,
        "score": float(score_new)
    }