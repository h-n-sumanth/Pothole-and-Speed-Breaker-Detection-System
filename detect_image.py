# detect_image.py
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = 'models/road_detection_model.h5'
FRAME_SIZE = (224, 224)
CONF_THRESHOLD = 0.55
labels = ['normal', 'pothole', 'speedbreaker']  # set to match training mapping

print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded")

def predict_image(path):
    img = load_img(path, target_size=FRAME_SIZE)
    x = img_to_array(img).astype('float32') / 255.0
    x = np.expand_dims(x, 0)
    pred = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    label = labels[idx]
    return label, conf, pred

def show_and_alert(path):
    label, conf, probs = predict_image(path)
    print(f"{os.path.basename(path)} -> {label} (confidence {conf:.2f})")
    print("All probs:", np.array2string(probs, precision=3, separator=', '))
    im = cv2.imread(path)
    cv2.putText(im, f"{label} {conf:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
    cv2.imshow("Single Image Test", im)
    if label != 'normal' and conf >= CONF_THRESHOLD:
        # Voice removed: just print to console instead
        print(f"[ALERT] {label} ahead (voice disabled)")
    else:
        print("No alert (normal or low confidence)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_dir = 'test_images'
    if not os.path.isdir(test_dir):
        print("Create folder 'test_images/' and copy your phone images there.")
        exit(1)
    files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not files:
        print("No images found in test_images/. Put phone photos there.")
        exit(1)
    for p in files:
        print("----")
        show_and_alert(p)
