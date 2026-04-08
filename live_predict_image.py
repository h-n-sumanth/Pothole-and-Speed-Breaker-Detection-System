import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/road_detection_model.h5', compile=False)

labels = ['normal', 'pothole', 'speedbreaker']
CONF_THRESHOLD = 0.60

cap = cv2.VideoCapture(0)

print("Camera started. Press 'q' to quit.")

def draw_colored_label(frame, text, bg_color, text_color=(0,0,0)):
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = 20
    pad_y = 12

    rect_w = tw + pad_x*2
    rect_h = th + pad_y*2

    x1 = int((w - rect_w)/2)
    y1 = 20

    x2 = x1 + rect_w
    y2 = y1 + rect_h

    cv2.rectangle(frame,(x1,y1),(x2,y2),bg_color,-1)

    text_x = x1 + pad_x
    text_y = y1 + pad_y + th

    cv2.putText(frame,text,(text_x,text_y),font,font_scale,text_color,thickness,cv2.LINE_AA)


prediction_started = False


while True:

    ret, frame = cap.read()

    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = np.expand_dims(img.astype('float32') / 255.0, axis=0)

    pred = model.predict(x,verbose=0)[0]

    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    label = labels[idx]


    if conf >= CONF_THRESHOLD:
        prediction_started = True


    if prediction_started:

        if label == 'normal':

            show_text = "Normal Road"
            bg = (0,200,0)
            txt = (0,0,0)

        elif label == 'pothole':

            show_text = "POTHOLE"
            bg = (0,0,200)
            txt = (0,0,0)

        else:

            show_text = "SPEEDBREAKER"
            bg = (0,220,220)
            txt = (0,0,0)

        draw_colored_label(frame,f"{show_text}  {conf:.2f}",bg,txt)

    cv2.imshow("Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

print("Stopped.")