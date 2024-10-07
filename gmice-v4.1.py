import cv2
import mediapipe as mp
import math
import ctypes
import time
import warnings
import collections
import tkinter as tk
from tkinter import ttk

# Suppress specific protobuf warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

class AirMouseController:
    def __init__(self):
        self.is_active = False
        self.tracking_started = False
        self.previous_hand_center = None
        self.hand_center_history = collections.deque(maxlen=10)
        
        # Initialize MediaPipe Hand model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configuration parameters
        self.sensitivity = 1.0
        self.dead_zone = 0.01
        self.smoothing_factor = 0.1
        self.scaling_factor = 0.5
        self.finger_distance_threshold = 0.1  # Distance threshold for stopping movement
        
        # Click timing
        self.click_cooldown = 0.3
        self.last_left_click_time = 0
        self.last_right_click_time = 0
        
        # Create GUI window
        self.create_gui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Air Mouse Controller")
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Toggle button
        self.toggle_button = ttk.Button(
            main_frame, 
            text="Start Air Mouse", 
            command=self.toggle_air_mouse
        )
        self.toggle_button.grid(row=0, column=0, pady=5)
        
        # Sensitivity slider
        ttk.Label(main_frame, text="Sensitivity:").grid(row=1, column=0, pady=2)
        self.sensitivity_slider = ttk.Scale(
            main_frame, 
            from_=0.1, 
            to=2.0, 
            orient=tk.HORIZONTAL, 
            value=self.sensitivity,
            command=self.update_sensitivity
        )
        self.sensitivity_slider.grid(row=2, column=0, pady=2)
        
        # Distance threshold slider
        ttk.Label(main_frame, text="Stop Distance:").grid(row=3, column=0, pady=2)
        self.distance_slider = ttk.Scale(
            main_frame, 
            from_=0.05, 
            to=0.2, 
            orient=tk.HORIZONTAL, 
            value=self.finger_distance_threshold,
            command=self.update_distance_threshold
        )
        self.distance_slider.grid(row=4, column=0, pady=2)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def toggle_air_mouse(self):
        self.is_active = not self.is_active
        self.toggle_button.config(
            text="Stop Air Mouse" if self.is_active else "Start Air Mouse"
        )
    
    def update_sensitivity(self, value):
        self.sensitivity = float(value)
    
    def update_distance_threshold(self, value):
        self.finger_distance_threshold = float(value)
    
    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def get_screen_resolution(self):
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    def move_mouse_rel(self, x, y):
        extra = ctypes.c_ulong(0)
        ii = MOUSEINPUT(x, y, 0, MOUSE_MOVE, 0, ctypes.pointer(extra))
        input_struct = INPUT(ctypes.c_ulong(0), _INPUT(mi=ii))
        ctypes.windll.user32.SendInput(1, ctypes.pointer(input_struct), ctypes.sizeof(INPUT))

    def distance(self, point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def is_fingers_touching(self, finger1, finger2):
        return self.distance(finger1, finger2) < self.finger_distance_threshold

    def get_hand_center(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        thumb_base = hand_landmarks.landmark[2]
        pinky_base = hand_landmarks.landmark[17]
        center_x = (wrist.x + thumb_base.x + pinky_base.x) / 3
        center_y = (wrist.y + thumb_base.y + pinky_base.y) / 3
        return center_x, center_y

    def calculate_smoothed_velocity(self, current_center):
        self.hand_center_history.append(current_center)
        
        if len(self.hand_center_history) > 1:
            avg_center_x = sum([center[0] for center in self.hand_center_history]) / len(self.hand_center_history)
            avg_center_y = sum([center[1] for center in self.hand_center_history]) / len(self.hand_center_history)
            
            delta_x = current_center[0] - avg_center_x
            delta_y = current_center[1] - avg_center_y
            
            return delta_x, delta_y
        return 0, 0

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        
        if result.multi_hand_landmarks and self.is_active:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = hand_landmarks.landmark[4]
                index_finger_tip = hand_landmarks.landmark[8]
                
                # Only stop tracking if fingers are far apart
                finger_distance = self.distance(thumb_tip, index_finger_tip)
                
                if finger_distance < self.finger_distance_threshold:
                    if not self.tracking_started:
                        self.tracking_started = True
                        self.previous_hand_center = self.get_hand_center(hand_landmarks)
                        self.hand_center_history.clear()
                    else:
                        current_hand_center = self.get_hand_center(hand_landmarks)
                        delta_x, delta_y = self.calculate_smoothed_velocity(current_hand_center)
                        
                        screen_width, screen_height = self.get_screen_resolution()
                        move_x = int(delta_x * screen_width * self.sensitivity)
                        move_y = int(delta_y * screen_height * self.sensitivity)
                        
                        if abs(move_x) > self.dead_zone or abs(move_y) > self.dead_zone:
                            self.move_mouse_rel(move_x, move_y)
                        
                        self.previous_hand_center = current_hand_center
                elif finger_distance > self.finger_distance_threshold * 2:  # Only stop if fingers are significantly apart
                    self.tracking_started = False
                    self.previous_hand_center = None
                    self.hand_center_history.clear()
        
        cv2.imshow("Air Mouse Controller", frame)

    def run(self):
        while True:
            self.process_frame()
            self.root.update()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Define necessary structures for mouse control
PUL = ctypes.POINTER(ctypes.c_ulong)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

class _INPUT(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUT)]

# Constants
MOUSE_MOVE = 0x0001

if __name__ == "__main__":
    controller = AirMouseController()
    controller.run()
