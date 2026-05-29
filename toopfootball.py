"""
این کد توسط امیرفرخانی موسس کدراه پیاده سازی شده
همکار و اساتید (تقلبی) قبل از برداشتن سورس کد
و تموم کردن کار به اسم خودتون ذکر منبع کنید
در غیر اینصورت رضایتی در کار نیست و اگه متوجه بشیم عواقب داره
این کد اولین باره داخل اینترنت پخش میشه
ویدیو مشابه داخل سطح نت ببینیم و ذکر منبع نکرده باشین عواقبش پای خودتون
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import WindowProperties, DirectionalLight, AmbientLight, Vec4, Texture, CardMaker
import cv2
import mediapipe as mp
import threading
import time
import sys
import signal
from PIL import Image, ImageSequence

class EyeTracker(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # تنظیمات پنجره
        props = WindowProperties()
        props.setCursorHidden(False)
        self.win.requestProperties(props)


        # نورپردازی
        dl = DirectionalLight("dl")
        dl.setColor(Vec4(0.5, 0.5, 0.5, 1))
        np = self.render.attachNewNode(dl)
        np.setPos(50, 50, 50)
        np.lookAt(0, 0, 0)
        self.render.setLight(np)

        al = AmbientLight("al")
        al.setColor(Vec4(0.5, 0.5, 0.55, 1))
        self.render.setLight(self.render.attachNewNode(al))

        # پس‌زمینه گیف روی render (تمام صفحه)
        try:
            gif = Image.open("giphy.gif")
            self.gif_frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]
            self.current_gif_index = 0

            self.bg_tex = Texture()
            cm_bg = CardMaker('bg_card')
            cm_bg.setFrame(-20, 20, -15, 15)  # بزرگ و تمام صفحه
            self.bg_card = self.render.attachNewNode(cm_bg.generate())
            self.bg_card.setPos(0, 10, 0)      # پشت توپ
            self.bg_card.setTexture(self.bg_tex)
        except Exception as e:
            print("گیف بارگذاری نشد:", e)
            self.gif_frames = None

        # توپ روی render
        self.eye = self.loader.loadModel("soccer_ball (1).glb")
        self.eye.reparentTo(self.render)
        self.eye.setScale(2.8)
        self.eye.setPos(0, 0, 0)

        # دوربین
        self.disableMouse()
        self.camera.setPos(0, -38, 3)
        self.camera.setHpr(0, -6, 0)
        self.camera.lookAt(0, 0, 1)

        # حرکت توپ
        self.current_x = 0.0
        self.current_z = 0.0
        self.smoothing = 0.12
        self.step_blink = 3.5
        self.step_hand = 1.0
        self.limit_x = 12.0
        self.limit_z = 6.0
        self.face_lock = threading.Lock()
        self.running = True
        self.blink_threshold = 0.26
        self.last_blink_time = 0.0
        self.left_wink_frames = 0
        self.right_wink_frames = 0
        self.both_closed_frames = 0
        self.min_wink_frames = 3
        self.last_hand_y = None

        # تکسچر وب‌کم بزرگ‌تر گوشه بالا سمت راست
        self.webcam_tex = Texture()
        cm = CardMaker('webcam_card')
        cm.setFrame(-0.4, 0.4, -0.3, 0.3)  # بزرگ‌تر
        self.webcam_card = self.aspect2d.attachNewNode(cm.generate())
        self.webcam_card.setPos(1 - 0, 0, 1 - 0.3)  # گوشه بالا سمت راست
        self.webcam_card.setTexture(self.webcam_tex)

        # شروع رشته‌ها
        threading.Thread(target=self.webcam_loop, daemon=True).start()
        self.taskMgr.add(self.update_eye, "update_eye")
        self.taskMgr.add(self.animate_gif_task, "animate_gif_task")

        signal.signal(signal.SIGINT, self._handle_exit)

    def _handle_exit(self, signum, frame):
        self.running = False
        try:
            self.userExit()
        except Exception:
            pass
        sys.exit(0)

    # انیمیشن گیف پس‌زمینه
    def animate_gif_task(self, task):
        if self.gif_frames:
            frame = self.gif_frames[self.current_gif_index]
            frame_rgb = frame.tobytes()
            self.bg_tex.setXSize(frame.width)
            self.bg_tex.setYSize(frame.height)
            self.bg_tex.setFormat(Texture.FRgb8)
            self.bg_tex.setRamImage(frame_rgb)
            self.current_gif_index = (self.current_gif_index + 1) % len(self.gif_frames)
        return task.again

    # وب‌کم
    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("دوربین باز نشد!")
            self.running = False
            return

        cap.set(3, 640)
        cap.set(4, 480)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.6, min_tracking_confidence=0.6
        )
        hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def ear(eye, lm):
            v1 = abs(lm.landmark[eye[1]].y - lm.landmark[eye[5]].y)
            v2 = abs(lm.landmark[eye[2]].y - lm.landmark[eye[4]].y)
            h = abs(lm.landmark[eye[0]].x - lm.landmark[eye[3]].x)
            return (v1 + v2) / (2.0 * (h + 1e-6))

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # تشخیص دست
            hand_results = hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                hand_y = hand_landmarks.landmark[0].y
                if self.last_hand_y is not None:
                    delta = hand_y - self.last_hand_y
                    with self.face_lock:
                        if delta < -0.005:
                            self.current_z = min(self.current_z + self.step_hand, self.limit_z)
                        elif delta > 0.005:
                            self.current_z = max(self.current_z - self.step_hand, -self.limit_z)
                self.last_hand_y = hand_y
            else:
                self.last_hand_y = None

            # تشخیص چشمک
            results = face_mesh.process(rgb)
            current_time = time.time()
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                left_ear = ear(LEFT_EYE, lm)
                right_ear = ear(RIGHT_EYE, lm)
                left_closed = left_ear < self.blink_threshold
                right_closed = right_ear < self.blink_threshold

                if left_closed and right_closed:
                    self.both_closed_frames += 1
                    self.left_wink_frames = 0
                    self.right_wink_frames = 0
                else:
                    if left_closed and not right_closed:
                        self.left_wink_frames += 1
                        self.right_wink_frames = 0
                        self.both_closed_frames = 0
                    elif right_closed and not left_closed:
                        self.right_wink_frames += 1
                        self.left_wink_frames = 0
                        self.both_closed_frames = 0
                    else:
                        self.left_wink_frames = 0
                        self.right_wink_frames = 0
                        self.both_closed_frames = 0

                can_wink = (self.both_closed_frames == 0)
                if can_wink and (current_time - self.last_blink_time > 0.45):
                    if self.left_wink_frames >= self.min_wink_frames:
                        with self.face_lock:
                            self.current_x = max(self.current_x - self.step_blink, -self.limit_x)
                        self.last_blink_time = current_time
                        self.left_wink_frames = 0
                    elif self.right_wink_frames >= self.min_wink_frames:
                        with self.face_lock:
                            self.current_x = min(self.current_x + self.step_blink, self.limit_x)
                        self.last_blink_time = current_time
                        self.right_wink_frames = 0

            # آپدیت وب‌کم روی تکسچر خودش
            cv2.flip(frame, 0, frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.webcam_tex.setXSize(frame_rgb.shape[1])
            self.webcam_tex.setYSize(frame_rgb.shape[0])
            self.webcam_tex.setFormat(Texture.FRgb8)
            self.webcam_tex.setRamImage(frame_rgb.tobytes())

            time.sleep(0.015)

        cap.release()
        face_mesh.close()
        hands.close()

    def update_eye(self, task):
        with self.face_lock:
            target_x = max(min(self.current_x, self.limit_x), -self.limit_x)
            target_z = max(min(self.current_z, self.limit_z), -self.limit_z)

        pos = self.eye.getPos()
        new_x = pos.x + (target_x - pos.x) * self.smoothing
        new_z = pos.z + (target_z - pos.z) * self.smoothing
        self.eye.setPos(max(min(new_x, self.limit_x), -self.limit_x), 0,
                        max(min(new_z, self.limit_z), -self.limit_z))
        return task.cont

if __name__ == "__main__":
    app = EyeTracker()
    app.run()
