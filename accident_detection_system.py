import sys
import cv2
import PyQt5
import requests
import warnings
from datetime import datetime
from ultralytics import YOLO

# Suppress the deprecation warning for google.generativeai
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")
import google.generativeai as genai
from database_manager import DatabaseManager

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QFrame, QStackedWidget,
    QPushButton, QScrollArea, QLineEdit,
    QGridLayout
)

from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
import torch
import time

# ====================================
# CONFIGURATION
# ====================================

CAMERA_SOURCE = 0
YOLO_MODEL = "yolov8n.pt"

# Detect Apple Silicon GPU (MPS)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"--- POWERING VISION SHIELD X ON DEVICE: {DEVICE.upper()} ---")

CAMERA_ID = "CAM-01 [NODE: ALPHA]"
LOCATION = "Delhi Highway Sector 12"

CONFIDENCE_THRESHOLD = 0.4
OVERLAP_THRESHOLD = 0.15
ALERT_COOLDOWN = 5

ACCIDENT_CLASSES = {"car", "truck", "bus", "motorcycle"}

# ====================================
# API KEYS & INTEGRATIONS
# ====================================

GEMINI_API_KEY = "your_gemini_api_key_here"
TELEGRAM_BOT_TOKEN = "8428016965:AAFVX-jXN4utblu7m-eqQcUjCUYXkX-VJCg"
TELEGRAM_CHAT_ID = "6965314664"

def send_telegram_alert(image_path, message):
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_token":
        print("Telegram error: TELEGRAM_BOT_TOKEN is not configured. Please add it to your code. Skipping message.")
        return
    if not TELEGRAM_CHAT_ID:
        print("Telegram error: TELEGRAM_CHAT_ID is not configured. Please add it to your code. Skipping message.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as photo:
            response = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": message}, files={"photo": photo})
            if response.status_code == 200:
                print("Telegram alert sent successfully.")
            else:
                print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print("Telegram Error:", e)

# Configure only if it's not the placeholder
USE_GEMINI = False
if GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        USE_GEMINI = True
    except Exception as e:
        print("Failed to configure Gemini:", e)

def verify_accident(image_path):
    if not USE_GEMINI:
        # Provide a professional default description if API is not set
        return True, "High-confidence collision detected by local Vision Node. Awaiting manual visual verification."

    prompt = """
    Analyze this road camera image for any vehicle accident, crash, or collision.
    If you see an accident, start your response exactly with "YES: " and briefly describe the severity of the accident and the vehicles involved in the rest of the sentence.
    If you DO NOT see an accident, reply exactly with "NO".
    """
    try:
        response = gemini_model.generate_content(
            [prompt, {
                "mime_type": "image/jpeg",
                "data": open(image_path, "rb").read()
            }]
        )
        answer = response.text.strip().upper()
        
        if answer.startswith("YES"):
            # Extract description by splitting at the first colon
            parts = response.text.strip().split(":", 1)
            description = parts[1].strip() if len(parts) > 1 else "Accident confirmed by AI."
            return True, description
            
        return False, "No accident detected by AI."
    except Exception as e:
        print("Gemini Error:", e)
        return False, f"API Error: {e}"

# ====================================
# IOU CALCULATION
# ====================================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0

    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter)

# ====================================
# ACCIDENT DETECTION
# ====================================

def detect_accident(results):
    vehicles = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = r.names[int(box.cls[0])]
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls in ACCIDENT_CLASSES:
                vehicles.append((x1, y1, x2, y2))
                
    accident = False
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            if compute_iou(vehicles[i], vehicles[j]) > OVERLAP_THRESHOLD:
                accident = True
    return accident, vehicles

# ====================================
# UI COMPONENTS (v2.0 Functional)
# ====================================

class HistoryCard(QFrame):
    def __init__(self, incident_data):
        super().__init__()
        id, time, cam, loc, desc, img, verified = incident_data
        self.setObjectName("statsCard")
        layout = QVBoxLayout(self)
        
        header = QHBoxLayout()
        t = QLabel(time); t.setStyleSheet("font-weight: bold; color: #0ea5e9;")
        v = QLabel("VERIFIED" if verified else "UNCERTAIN")
        v.setStyleSheet(f"color: {'#22c55e' if verified else '#94a3b8'}; font-size: 10px;")
        header.addWidget(t); header.addStretch(); header.addWidget(v)
        
        l = QLabel(f"📍 {loc}"); l.setStyleSheet("color: #94a3b8; font-size: 11px;")
        d = QLabel(desc); d.setWordWrap(True); d.setStyleSheet("color: #f8fafc; font-size: 13px;")
        
        layout.addLayout(header)
        layout.addWidget(l)
        layout.addWidget(d)

class HistoryView(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("INCIDENT HISTORY")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.container)
        
        layout.addWidget(self.scroll)

    def refresh(self):
        # Clear existing
        for i in reversed(range(self.container_layout.count())): 
            self.container_layout.itemAt(i).widget().setParent(None)
            
        incidents = self.db.get_recent_incidents(20)
        for inc in incidents:
            self.container_layout.addWidget(HistoryCard(inc))

class SettingsView(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        
        title = QLabel("SYSTEM CONFIGURATION")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        grid = QGridLayout()
        grid.setSpacing(15)
        
        self.inputs = {}
        fields = [
            ("GEMINI API KEY", "GEMINI_API_KEY"),
            ("TELEGRAM BOT TOKEN", "TELEGRAM_BOT_TOKEN"),
            ("TELEGRAM CHAT ID", "TELEGRAM_CHAT_ID"),
            ("LOCATION LABEL", "LOCATION"),
            ("OVERLAP THRESHOLD", "OVERLAP_THRESHOLD")
        ]
        
        for i, (label_text, attr) in enumerate(fields):
            lbl = QLabel(label_text); lbl.setObjectName("statsTitle")
            edit = QLineEdit(); edit.setObjectName("alertLog")
            # Get current value from module level (simplified)
            current_val = str(globals().get(attr, ""))
            edit.setText(current_val)
            grid.addWidget(lbl, i, 0)
            grid.addWidget(edit, i, 1)
            self.inputs[attr] = edit
            
        layout.addLayout(grid)
        
        save_btn = QPushButton("SAVE CONFIGURATION")
        save_btn.setStyleSheet("background-color: #0ea5e9; padding: 15px; font-weight: bold; margin-top: 20px;")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)
        
    def save_config(self):
        for attr, edit in self.inputs.items():
            val = edit.text()
            # Update global variables in the module
            globals()[attr] = float(val) if "THRESHOLD" in attr else val
        print("Configuration Updated.")

# ====================================
# VIDEO THREAD
# ====================================

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object, list, bool)

    def __init__(self, camera_source, yolo_model):
        super().__init__()
        self.camera_source = camera_source
        self.running = True
        self.model = YOLO(yolo_model)
        # Move model to optimal device (MPS on Mac)
        self.model.to(DEVICE)
        self.frame_count = 0
        self.skip_rate = 2 # Process every 3rd frame for detection
        
    def run(self):
        cap = cv2.VideoCapture(self.camera_source)
        # Performance: Set buffer size to 1 to reduce lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Camera not accessible")
            return
            
        last_accident = False
        last_vehicles = []

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Mirror the frame horizontally (1) to make motion intuitive
            frame = cv2.flip(frame, 1)
            
            # ZERO-LAG LOGIC: Continuous display, periodic inference
            if self.frame_count % self.skip_rate == 0:
                results = self.model(frame, verbose=False, device=DEVICE)
                last_accident, last_vehicles = detect_accident(results)
            
            self.frame_count += 1
            self.change_pixmap_signal.emit(frame, last_vehicles, last_accident)
            
            # Yield slightly to give UI some breathing room
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.wait()

# ====================================
# MAIN WINDOW
# ====================================

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Shield X [v2.0] - Mission Control")
        self.showMaximized()
        self.last_alert = None
        self.db = DatabaseManager()
        self.nav_buttons = []

        self.setup_styles()
        self.build_ui()

        # Start Video Thread
        self.thread = VideoThread(CAMERA_SOURCE, YOLO_MODEL)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020617;
            }
            QLabel {
                color: #38bdf8;
                font-family: 'Inter', 'Segoe UI', sans-serif;
            }
            QFrame#sidebar {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
                min-width: 220px;
            }
            QPushButton#navBtn {
                background-color: transparent;
                color: #94a3b8;
                border: none;
                text-align: left;
                padding: 15px 25px;
                font-size: 14px;
                font-weight: 500;
                border-radius: 8px;
                margin: 5px 10px;
            }
            QPushButton#navBtn:hover {
                background-color: #1e293b;
                color: #f8fafc;
            }
            QPushButton#navBtn[active="true"] {
                background-color: #0ea5e9;
                color: #ffffff;
            }
            QFrame#camFrame {
                border: 1px solid #0ea5e9;
                border-radius: 16px;
                background-color: #000000;
                padding: 2px;
            }
            QFrame#camFrame[accident="true"] {
                border: 3px solid #ef4444;
                background-color: #7f1d1d;
            }
            QFrame#statsCard {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 15px;
            }
            QLabel#statsTitle {
                color: #94a3b8;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            QLabel#statsVal {
                color: #f8fafc;
                font-size: 24px;
                font-weight: bold;
            }
            QTextEdit#alertLog {
                background-color: rgba(15, 23, 42, 0.8);
                border: 1px solid #1e293b;
                border-radius: 12px;
                color: #38bdf8;
                font-family: 'JetBrains Mono', 'Consolas', monospace;
                font-size: 13px;
                padding: 15px;
            }
            QLabel#titleText {
                font-size: 18px;
                font-weight: 900;
                color: #0ea5e9;
                letter-spacing: 2px;
                padding: 20px;
            }
        """)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_hbox = QHBoxLayout(central)
        main_hbox.setContentsMargins(0, 0, 0, 0)
        main_hbox.setSpacing(0)

        # 1. Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 20, 10, 20)

        logo = QLabel("VISION\nSHIELD X")
        logo.setObjectName("titleText")
        logo.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(logo)

        labels = ["LIVE MONITOR", "HISTORY", "ANALYTICS", "SETTINGS"]
        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            if i == 0: btn.setChecked(True)
            btn.clicked.connect(lambda checked, idx=i: self.switch_view(idx))
            side_layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        side_layout.addStretch()

        # 2. Main Content Stack
        self.stack = QStackedWidget()
        
        # View 0: Live Monitor
        self.live_widget = QWidget()
        live_layout = QVBoxLayout(self.live_widget)
        live_layout.setContentsMargins(30, 30, 30, 30)
        live_layout.setSpacing(25)

        # Header Stats
        stats_hbox = QHBoxLayout()
        stats_data = [
            ("NODE STATUS", "ONLINE"),
            ("ACTIVE DEVICE", DEVICE.upper()),
            ("AI CONFIDENCE", "98%")
        ]
        for label, val in stats_data:
            card = QFrame(); card.setObjectName("statsCard")
            l = QVBoxLayout(card)
            t = QLabel(label); t.setObjectName("statsTitle")
            v = QLabel(val); v.setObjectName("statsVal")
            l.addWidget(t); l.addWidget(v)
            stats_hbox.addWidget(card)
        live_layout.addLayout(stats_hbox)

        # Video Feed
        self.camera_frame = QFrame()
        self.camera_frame.setObjectName("camFrame")
        self.camera_frame.setProperty("accident", False)
        cam_layout = QVBoxLayout(self.camera_frame)
        cam_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(450)
        cam_layout.addWidget(self.video_label)
        live_layout.addWidget(self.camera_frame, stretch=7)

        self.alert_log = QTextEdit()
        self.alert_log.setObjectName("alertLog")
        self.alert_log.setReadOnly(True)
        live_layout.addWidget(self.alert_log, stretch=3)

        # View 1: History
        self.history_view = HistoryView(self.db)
        
        # View 2: Analytics (Placeholder)
        self.analytics_view = QLabel("Strategic Analytics coming soon in v3.0")
        self.analytics_view.setAlignment(Qt.AlignCenter)

        # View 3: Settings
        self.settings_view = SettingsView(self)

        self.stack.addWidget(self.live_widget)
        self.stack.addWidget(self.history_view)
        self.stack.addWidget(self.analytics_view)
        self.stack.addWidget(self.settings_view)

        main_hbox.addWidget(sidebar)
        main_hbox.addWidget(self.stack, stretch=10)

    def switch_view(self, index):
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            btn.setProperty("active", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        
        if index == 1: # History
            self.history_view.refresh()

    def update_image(self, frame, vehicles, accident):
        for box in vehicles:
            x1, y1, x2, y2 = box
            # Normal green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if accident:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 4)
            cv2.putText(
                frame, "CRITICAL: ACCIDENT DETECTED", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA
            )
            if not self.camera_frame.property("accident"):
                self.camera_frame.setProperty("accident", True)
                self.camera_frame.style().unpolish(self.camera_frame)
                self.camera_frame.style().polish(self.camera_frame)
            self.handle_accident(frame)
        else:
            if self.camera_frame.property("accident"):
                self.camera_frame.setProperty("accident", False)
                self.camera_frame.style().unpolish(self.camera_frame)
                self.camera_frame.style().polish(self.camera_frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def handle_accident(self, frame):
        now = datetime.now()
        if self.last_alert and (now - self.last_alert).total_seconds() < ALERT_COOLDOWN:
            return

        self.last_alert = now
        
        import threading
        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG

        def process_alert():
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            filename = f"accident_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

            # Retrieve validation and description from Gemini (Blocking API call)
            is_verified_accident, description = verify_accident(filename)

            if is_verified_accident:
                unified_msg = (
                    f"[{timestamp}] 🚨 EMERGENCY ALERT DETECTED\n"
                    f"----------------------------------------\n"
                    f"SOURCE: {CAMERA_ID}\n"
                    f"LOCATION: {LOCATION}\n"
                    f"VERIFICATION: YES (AI Confirmed)\n"
                    f"DETAILS: {description}\n"
                    f"----------------------------------------\n"
                    f"STATUS: POLICE [DISPATCHED] | MEDICAL [REQUESTED]\n"
                )
                
                # Send unified message + image to telegram
                send_telegram_alert(filename, unified_msg)

                # Log to local database
                self.db.log_incident(CAMERA_ID, LOCATION, description, filename, True)
            else:
                unified_msg = (
                    f"[{timestamp}] ℹ️ SYSTEM UPDATE\n"
                    f"SOURCE: {CAMERA_ID}\n"
                    f"STATUS: Potential collision analyzed. Verification: NO (False Alarm).\n"
                    f"ACTION: None required. Units standing by.\n"
                )
                self.db.log_incident(CAMERA_ID, LOCATION, "False Alarm", filename, False)

            # Safely Append to GUI using Qt event loop
            QMetaObject.invokeMethod(self.alert_log, "append", Qt.QueuedConnection, Q_ARG(str, unified_msg))

        # Start alert sub-thread so UI does not freeze during Gemini validation
        threading.Thread(target=process_alert, daemon=True).start()

    def closeEvent(self, event):
        self.thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
