import os
import time
import cv2
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import NW
from PIL import Image, ImageTk

# ---------------- CONFIG (detection logic unchanged) ----------------
MODEL_PATH = 'models/road_detection_model.h5'
FRAME_SIZE = (224, 224)
SMOOTH_WINDOW = 5
CONF_THRESHOLD = 0.70
CONSECUTIVE_REQUIRED = 3
ALERT_DISPLAY_TIME = 3.0
AUTO_PRED_INTERVAL = 0.2

labels = ['normal', 'pothole', 'speedbreaker']

MIN_PHONE_AREA_RATIO = 0.08
MAX_PHONE_APPROX_EPS = 0.06

# Optional menu background (put an image at assets/menu_bg.jpg)
MENU_BG = os.path.join('assets', 'menu_bg.jpg')

# Preview size shown on screen
PREVIEW_W, PREVIEW_H = 960, 540

# ---------------- load model once ----------------
print("Loading model:", MODEL_PATH)
model = load_model(MODEL_PATH, compile=False)
print("Model loaded. Input shape:", model.input_shape)


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_phone_quad(gray, frame):
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    frame_area = w * h
    for cnt in contours:
        if cv2.contourArea(cnt) < frame_area * MIN_PHONE_AREA_RATIO:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, MAX_PHONE_APPROX_EPS * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            try:
                warped = four_point_transform(frame, pts)
                if warped.size < 32 * 32:
                    continue
                return warped, pts
            except Exception:
                continue
    return None, None


def preprocess(roi):
    img = cv2.resize(roi, FRAME_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img.astype("float32") / 255.0, 0)


def predict_roi(roi):
    pred = model.predict(preprocess(roi), verbose=0)[0]
    idx = int(np.argmax(pred))
    return labels[idx], float(pred[idx]), pred


# ---------------- Live detector class (wraps detection loop) ----------------

class LiveDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.history = deque(maxlen=SMOOTH_WINDOW if SMOOTH_WINDOW > 0 else 1)
        self.candidate_label = None
        self.candidate_count = 0
        self.last_detected_label = None
        self.last_detected_conf = 0.0
        self.last_detected_time = 0.0
        self.next_auto_time = time.time()
        self.window_name = "Live Detection"
        self.exit_btn_coords = None  # (x1,y1,x2,y2) in preview coordinates

    def start_camera(self):
        # open camera and run loop
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam (device 0).")
        self.running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._loop()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyWindow(self.window_name)

    def _mouse_callback(self, event, x, y, flags, param):
        # mouse callback for preview window — if user clicks inside exit_btn_coords, stop running
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.exit_btn_coords:
                x1, y1, x2, y2 = self.exit_btn_coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.running = False

    def _draw_exit_button_on_preview(self, preview):
        # bottom-left exit button on preview image; returns coords tuple
        ex_w, ex_h = 100, 36
        ex_x1 = 10
        ex_y1 = PREVIEW_H - 10 - ex_h
        ex_x2 = ex_x1 + ex_w
        ex_y2 = ex_y1 + ex_h
        cv2.rectangle(preview, (ex_x1, ex_y1), (ex_x2, ex_y2), (60, 60, 60), -1)
        cv2.putText(preview, "EXIT", (ex_x1 + 16, ex_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return (ex_x1, ex_y1, ex_x2, ex_y2)

    def _loop(self):
        print("Camera started. Click EXIT (bottom-left) or press 'q' to return to Start Menu.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            draw = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            warped, quad = detect_phone_quad(gray, frame)
            phone_present = warped is not None

            now = time.time()
            do_predict = now >= self.next_auto_time

            if phone_present:
                pts = quad.reshape(4, 2).astype(int)
                cv2.polylines(draw, [pts], True, (0, 255, 0), 2)
                try:
                    draw[10:170, 10:250] = cv2.resize(warped, (240, 160))
                except Exception:
                    pass

                if do_predict:
                    label, conf, probs = predict_roi(warped)
                    self.history.append((label, conf))
                    votes = [h[0] for h in self.history]
                    most_common = Counter(votes).most_common(1)[0][0]
                    confs = [h[1] for h in self.history if h[0] == most_common]
                    avg_conf = float(np.mean(confs)) if confs else conf

                    # same candidate acceptance logic
                    if most_common != "normal" and avg_conf >= CONF_THRESHOLD:
                        if self.candidate_label == most_common:
                            self.candidate_count += 1
                        else:
                            self.candidate_label = most_common
                            self.candidate_count = 1
                    else:
                        self.candidate_label = None
                        self.candidate_count = 0

                    if self.candidate_label and self.candidate_count >= CONSECUTIVE_REQUIRED:
                        self.last_detected_label = self.candidate_label
                        self.last_detected_conf = avg_conf
                        self.last_detected_time = now
                        print("ACCEPTED ->", self.last_detected_label, self.last_detected_conf)
                        self.candidate_label = None
                        self.candidate_count = 0

                    self.next_auto_time = now + AUTO_PRED_INTERVAL
            else:
                cv2.putText(draw, "Show phone to detect",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 255), 2)

            # ----- UI BANNER: FULL-WIDTH to hide phone wallpaper artifacts -----
            banner_left = 0
            banner_top = 10
            banner_right = draw.shape[1]
            banner_bottom = banner_top + 56

            if self.last_detected_label and (time.time() - self.last_detected_time) <= ALERT_DISPLAY_TIME:
                if self.last_detected_label == "pothole":
                    text = f"POTHOLE  {self.last_detected_conf:.2f}"
                    bg = (0, 0, 0)
                    col = (0, 0, 255)
                    icon = "triangle"
                elif self.last_detected_label == "speedbreaker":
                    text = f"SPEEDBREAKER  {self.last_detected_conf:.2f}"
                    bg = (0, 0, 0)
                    col = (0, 200, 255)
                    icon = "circle"
                else:
                    text = "Normal Road"
                    bg = (0, 255, 0)
                    col = (0, 0, 0)
                    icon = "check"
            else:
                text = "Normal Road"
                bg = (0, 255, 0)
                col = (0, 0, 0)
                icon = "check"

            # draw full-width banner
            cv2.rectangle(draw, (banner_left, banner_top), (banner_right, banner_bottom), bg, -1)

            # icon box on left side of banner (keeps banner full width)
            icon_x = banner_left + 10
            icon_y = banner_top + 4
            cv2.rectangle(draw, (icon_x, icon_y), (icon_x + 56, icon_y + 48), (40, 40, 40), -1)

            if icon == "triangle":
                pts_icon = np.array([[icon_x + 8, icon_y + 40],
                                     [icon_x + 28, icon_y + 8],
                                     [icon_x + 48, icon_y + 40]])
                cv2.fillPoly(draw, [pts_icon], (0, 0, 255))
            elif icon == "circle":
                cv2.circle(draw, (icon_x + 28, icon_y + 28), 18, (0, 200, 255), -1)
            else:
                cv2.line(draw, (icon_x + 10, icon_y + 28), (icon_x + 26, icon_y + 44), (0, 200, 0), 6)
                cv2.line(draw, (icon_x + 26, icon_y + 44), (icon_x + 46, icon_y + 12), (0, 200, 0), 6)

            # center text inside banner
            size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)[0]
            tx = banner_left + (banner_right - banner_left - size[0]) // 2
            ty = banner_top + 42
            cv2.putText(draw, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4, cv2.LINE_AA)

            # draw preview and exit button (preview size PREVIEW_W x PREVIEW_H)
            preview = cv2.resize(draw, (PREVIEW_W, PREVIEW_H))

            # draw bottom-left exit button and set coords for callback
            ex_w, ex_h = 100, 36
            ex_x1 = 10
            ex_y1 = PREVIEW_H - 10 - ex_h
            ex_x2 = ex_x1 + ex_w
            ex_y2 = ex_y1 + ex_h
            cv2.rectangle(preview, (ex_x1, ex_y1), (ex_x2, ex_y2), (60, 60, 60), -1)
            cv2.putText(preview, "EXIT", (ex_x1 + 16, ex_y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # update exit button coords (window coords)
            # Important: OpenCV window shows the preview directly, so mouse coords correspond to preview coords
            self.exit_btn_coords = (ex_x1, ex_y1, ex_x2, ex_y2)

            cv2.imshow(self.window_name, preview)

            # allow keyboard exit too
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('e'):
                self.running = False

        # cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyWindow(self.window_name)
        print("Camera stopped; returning to Start Menu.")


# ---------------- Tkinter Start Menu UI ----------------

class StartMenuApp:
    def __init__(self, root):
        self.root = root
        root.title("Pothole & Speedbreaker Detector")
        root.geometry(f"{PREVIEW_W}x{PREVIEW_H}")
        root.resizable(False, False)

        # load optional background
        if os.path.exists(MENU_BG):
            img = Image.open(MENU_BG).resize((PREVIEW_W, PREVIEW_H))
            self.bg_img = ImageTk.PhotoImage(img)
        else:
            self.bg_img = None

        self.canvas = tk.Canvas(root, width=PREVIEW_W, height=PREVIEW_H, highlightthickness=0)
        self.canvas.pack()

        if self.bg_img:
            self.canvas.create_image(0, 0, anchor=NW, image=self.bg_img)
        else:
            # draw gradient background
            for y in range(PREVIEW_H):
                color = "#%02x%02x%02x" % (25 + int(120 * y / PREVIEW_H), 30 + int(60 * y / PREVIEW_H), 80 + int(80 * y / PREVIEW_H))
                self.canvas.create_line(0, y, PREVIEW_W, y, fill=color)

        # glass panel overlay
        panel_left, panel_top = 80, 80
        panel_right, panel_bottom = PREVIEW_W - 80, PREVIEW_H - 120
        self.canvas.create_rectangle(panel_left, panel_top, panel_right, panel_bottom, fill='#000000', stipple='gray50', outline='')

        # title and subtitle
        self.canvas.create_text(PREVIEW_W // 2, 140, text="Pothole & Speedbreaker Detector", font=("Helvetica", 28, "bold"), fill="white")
        self.canvas.create_text(PREVIEW_W // 2, 180, text="BE CAREFUL AND STAY SAFE ON ROAD", font=("Helvetica", 14), fill="#e6e6e6")

        # START button
        self.start_btn = tk.Button(root, text="START", font=("Arial", 16, "bold"), command=self.on_start,
                                   bg="#00a000", fg="white", padx=30, pady=12)
        self.start_win = self.canvas.create_window(PREVIEW_W // 2 - 140, PREVIEW_H // 2 + 20, window=self.start_btn)

        # QUIT button
        self.quit_btn = tk.Button(root, text="QUIT", font=("Arial", 16, "bold"), command=self.on_quit,
                                  bg="#c00000", fg="white", padx=30, pady=12)
        self.quit_win = self.canvas.create_window(PREVIEW_W // 2 + 140, PREVIEW_H // 2 + 20, window=self.quit_btn)

        self.detector = LiveDetector()

    def on_start(self):
        # hide menu and run detection; when detection stops, menu reappears
        self.root.withdraw()
        try:
            self.detector.start_camera()
        except Exception as e:
            print("Error starting camera:", e)
        self.root.deiconify()

    def on_quit(self):
        self.root.quit()


# ---------------- Main ----------------

def main():
    root = tk.Tk()
    app = StartMenuApp(root)
    root.mainloop()
    # ensure camera stopped
    try:
        app.detector.stop_camera()
    except Exception:
        pass
    print("Application exited.")


if __name__ == "__main__":
    main()
