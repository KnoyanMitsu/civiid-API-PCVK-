import cv2 as cv
import os

# Init paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '../models')

# Load Cascades
face_cascade = cv.CascadeClassifier(os.path.join(models_dir, 'haarcascade_frontalface_default.xml'))
left_eye_cascade = cv.CascadeClassifier(os.path.join(models_dir, 'haarcascade_mcs_lefteye.xml'))
right_eye_cascade = cv.CascadeClassifier(os.path.join(models_dir, 'haarcascade_mcs_righteye.xml'))
nose_cascade = cv.CascadeClassifier(os.path.join(models_dir, 'haarcascade_mcs_nose.xml'))
mouth_cascade = cv.CascadeClassifier(os.path.join(models_dir, 'haarcascade_mcs_mouth.xml'))

def detect_face_and_all_features(img_input):
    if isinstance(img_input, str):
        img = cv.imread(img_input)
    else:
        img = img_input

    if img is None:
        return None

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # DETEKSI WAJAH UTAMA
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_color = img[y:y+h, x:x+w]
    face_gray  = gray[y:y+h, x:x+w]

    H, W = face_gray.shape

    # ======================================
    # 1. DETEKSI MATA KIRI
    # ======================================
    left_roi = face_gray[0:int(H*0.50), 0:int(W*0.60)]
    left_eye = left_eye_cascade.detectMultiScale(left_roi, 1.1, 6)

    if len(left_eye) == 0:
        return None

    lex, ley, lew, leh = left_eye[0]
    lex_global, ley_global = lex, ley
    left_eye_crop = face_color[ley_global:ley_global+leh, lex_global:lex_global+lew]

    # ======================================
    # 2. DETEKSI MATA KANAN
    # ======================================
    right_roi = face_gray[0:int(H*0.50), int(W*0.40):W]
    right_eye = right_eye_cascade.detectMultiScale(right_roi, 1.1, 6)

    if len(right_eye) == 0:
        return None

    rex, rey, rew, reh = right_eye[0]
    rex_global = rex + int(W*0.40)
    rey_global = rey
    right_eye_crop = face_color[rey_global:rey_global+reh, rex_global:rex_global+rew]

    # ======================================
    # 3. DETEKSI HIDUNG
    # ======================================
    nose_roi_y1 = int(0.35 * H)
    nose_roi_y2 = int(0.70 * H)
    nose_roi_gray = face_gray[nose_roi_y1:nose_roi_y2, :]

    nose = nose_cascade.detectMultiScale(nose_roi_gray, 1.1, 6)
    if len(nose) == 0:
        return None

    nx, ny, nw, nh = nose[0]
    ny_global = ny + nose_roi_y1
    nose_crop = face_color[ny_global:ny_global+nh, nx:nx+nw]

    # ======================================
    # 4. DETEKSI MULUT
    # ======================================
    mouth_roi_y1 = int(0.65 * H)
    mouth_roi_y2 = int(0.95 * H)
    mouth_roi_gray = face_gray[mouth_roi_y1:mouth_roi_y2, :]

    mouth = mouth_cascade.detectMultiScale(mouth_roi_gray, 1.1, 6)
    if len(mouth) == 0:
        return None

    mx, my, mw, mh = mouth[0]
    my_global = my + mouth_roi_y1
    mouth_crop = face_color[my_global:my_global+mh, mx:mx+mw]

    # ======================================
    # SIMPAN HASIL
    # ======================================
    return {
        "face": face_color,
        "left_eye": left_eye_crop,
        "right_eye": right_eye_crop,
        "nose": nose_crop,
        "mouth": mouth_crop,
        "boxes": {
            "face": (0,0,w,h),
            "left_eye": (lex_global, ley_global, lew, leh),
            "right_eye": (rex_global, rey_global, rew, reh),
            "nose": (nx, ny_global, nw, nh),
            "mouth": (mx, my_global, mw, mh)
        }
    }