import cv2
import torch
import numpy as np
import pygame
import keyboard
import time
import win32gui
import win32con
import win32api
import ctypes
import warnings
from mss import mss
import random
from pynput.mouse import Listener
import json
import os
from collections import deque


warnings.filterwarnings("ignore", category=FutureWarning)


SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

# Default settings 
DEFAULT_SETTINGS = {
    "detection": {
        "model": {"value": "yolov5n", "description": "YOLOv5 model variant to use (e.g., yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)"},
        "min_confidence": {"value": 0.5, "description": "Minimum confidence score for a detection to be considered valid (0.0 to 1.0)"},
        "resize_width": {"value": 640, "description": "Width to resize screen capture for detection (smaller = faster, less accurate)"},
        "resize_height": {"value": 480, "description": "Height to resize screen capture for detection (smaller = faster, less accurate)"},
        "detection_radius": {"value": 300, "description": "Radius (in pixels) around screen center where detections are considered"},
        "min_box_area": {"value": 1000, "description": "Minimum area (width * height in pixels) of a bounding box to be valid"},
        "max_aspect_ratio": {"value": 3.0, "description": "Maximum width/height ratio of a bounding box to filter odd shapes (e.g., 3.0 means max 3:1 or 1:3)"}
    },
    "aiming": {
        "smooth_factor": {"value": 0.3, "description": "Fraction of remaining distance to move per frame (0.0 to 1.0, higher = faster)"},
        "jitter_min": {"value": -2, "description": "Minimum random jitter (in pixels) added to mouse movement for human-like behavior"},
        "jitter_max": {"value": 2, "description": "Maximum random jitter (in pixels) added to mouse movement"},
        "max_move_speed": {"value": 100, "description": "Maximum pixels to move per frame (caps movement speed)"},
        "movement_delay_min": {"value": 0.001, "description": "Minimum delay (seconds) between micro-movements for anti-cheat evasion"},
        "movement_delay_max": {"value": 0.003, "description": "Maximum delay (seconds) between micro-movements for anti-cheat evasion"},
        "interpolation_steps": {"value": 10, "description": "Number of steps for smooth mouse movement interpolation per detection (lower = faster)"},
        "movement_prediction": {
            "enabled": {"value": False, "description": "Enable AI-based movement prediction to track moving targets (True/False)"},
            "prediction_frames": {"value": 10, "description": "Number of frames to use for predicting opponent movement (higher = smoother but less responsive)"}
        }
    },
    "overlay": {
        "circle_color": {"value": [255, 0, 0], "description": "RGB color of the detection radius circle (e.g., [255, 0, 0] = red)"},
        "line_color": {"value": [0, 255, 0], "description": "RGB color of lines to detected targets (e.g., [0, 255, 0] = green)"},
        "box_color": {"value": [0, 0, 255], "description": "RGB color of bounding boxes around targets (e.g., [0, 0, 255] = blue)"},
        "text_color": {"value": [255, 255, 255], "description": "RGB color of confidence text (e.g., [255, 255, 255] = white)"},
        "font_size": {"value": 24, "description": "Font size for confidence text display"},
        "line_width": {"value": 1, "description": "Width (in pixels) of lines and bounding boxes"},
        "circle_width": {"value": 2, "description": "Width (in pixels) of the detection radius circle"},
        "overlay_alpha": {"value": 1.0, "description": "Transparency of the overlay (0.0 = fully transparent, 1.0 = fully opaque)"}
    },
    "performance": {
        "target_fps": {"value": 30, "description": "Desired frames per second for the aimbot loop"},
        "detection_interval": {"value": 1, "description": "Number of frames between detection updates (1 = every frame, higher = less frequent)"}
    },
    "controls": {
        "toggle_aimbot_key": {"value": "num 1", "description": "Key to toggle aimbot on/off (e.g., 'num 1', 'f1', 'space')"},
        "toggle_aim_only_key": {"value": "num 2", "description": "Key to toggle aim-only mode (requires right-click when enabled)"}
    },
    "debug": {
        "enable_debug": {"value": False, "description": "Enable debug output in the terminal (True/False)"}
    }
}


def merge_settings(defaults, current):
    merged = {}
    for category, settings_dict in defaults.items():
        merged[category] = {}
        if category not in current:
            merged[category] = settings_dict
        else:
            for key, default_item in settings_dict.items():
                if key in current[category]:
                    current_item = current[category][key]
                    if isinstance(default_item, dict) and "value" in default_item:
                       
                        if isinstance(current_item, dict) and "value" in current_item:
                            
                            merged[category][key] = {
                                "value": current_item["value"],
                                "description": default_item["description"]
                            }
                        else:
                            
                            merged[category][key] = {
                                "value": current_item,
                                "description": default_item["description"]
                            }
                    elif isinstance(default_item, dict):
                       
                        merged[category][key] = merge_settings(default_item, current_item)
                    else:
                        
                        merged[category][key] = current_item
                else:
                    
                    merged[category][key] = default_item
    return merged

# Load or create settings
if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(DEFAULT_SETTINGS, f, indent=4)
    print(f"Created new settings file: {SETTINGS_FILE}")
    settings = DEFAULT_SETTINGS
else:
    with open(SETTINGS_FILE, 'r') as f:
        current_settings = json.load(f)
    settings = merge_settings(DEFAULT_SETTINGS, current_settings)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"Settings loaded and updated: {SETTINGS_FILE}")


model = torch.hub.load('ultralytics/yolov5', settings["detection"]["model"]["value"], pretrained=True)


aimbot_enabled = False
aim_only_enabled = False
right_click_held = False


MOUSEEVENTF_MOVE = 0x0001


sct = mss()


def on_click(x, y, button, pressed):
    global right_click_held
    if button == button.right:
        right_click_held = pressed


listener = Listener(on_click=on_click)
listener.start()


def toggle_aimbot():
    global aimbot_enabled
    aimbot_enabled = not aimbot_enabled
    print(f"Aimbot toggled {'ON' if aimbot_enabled else 'OFF'}")


def toggle_aim_only():
    global aim_only_enabled
    aim_only_enabled = not aim_only_enabled
    print(f"Aim Only mode toggled {'ON' if aim_only_enabled else 'OFF'}")


keyboard.on_press_key(settings["controls"]["toggle_aimbot_key"]["value"], lambda _: toggle_aimbot())
keyboard.on_press_key(settings["controls"]["toggle_aim_only_key"]["value"], lambda _: toggle_aim_only())


def is_within_radius(x, y, center_x, center_y, radius):
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def detect_players(screen_width, screen_height):
    screenshot = np.array(sct.grab({"top": 0, "left": 0, "width": screen_width, "height": screen_height}))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
    frame = cv2.resize(frame, (settings["detection"]["resize_width"]["value"], settings["detection"]["resize_height"]["value"]))
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    scale_x, scale_y = screen_width / settings["detection"]["resize_width"]["value"], screen_height / settings["detection"]["resize_height"]["value"]
    center_x, center_y = screen_width // 2, screen_height // 2
    targets = []

    for det in detections:
        x_min, y_min, x_max, y_max, conf, cls = det
        if int(cls) == 0 and conf >= settings["detection"]["min_confidence"]["value"]:
            x_min, x_max = x_min * scale_x, x_max * scale_x
            y_min, y_max = y_min * scale_y, y_max * scale_y
            target_x = (x_min + x_max) / 2
            target_y = (y_min + y_max) / 2

            box_area = (x_max - x_min) * (y_max - y_min)
            aspect_ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else float('inf')
            if (box_area >= settings["detection"]["min_box_area"]["value"] and 
                1/settings["detection"]["max_aspect_ratio"]["value"] <= aspect_ratio <= settings["detection"]["max_aspect_ratio"]["value"] and 
                is_within_radius(target_x, target_y, center_x, center_y, settings["detection"]["detection_radius"]["value"])):
                overlay_x = target_x - (center_x - settings["detection"]["detection_radius"]["value"])
                overlay_y = target_y - (center_y - settings["detection"]["detection_radius"]["value"])
                targets.append({
                    "center": (overlay_x, overlay_y),
                    "box": (x_min - (center_x - settings["detection"]["detection_radius"]["value"]), 
                            y_min - (center_y - settings["detection"]["detection_radius"]["value"]),
                            x_max - (center_x - settings["detection"]["detection_radius"]["value"]), 
                            y_max - (center_y - settings["detection"]["detection_radius"]["value"])),
                    "conf": conf,
                    "screen_x": target_x,
                    "screen_y": target_y
                })
                if settings["debug"]["enable_debug"]["value"]:
                    print(f"Detected: ({target_x}, {target_y}), Conf: {conf}, Area: {box_area}, Aspect: {aspect_ratio}")

    if settings["debug"]["enable_debug"]["value"]:
        print(f"Targets detected: {len(targets)}")
    return targets


def predict_next_position(target_x, target_y, position_history, frame_time):
    if len(position_history) < 2:
        return target_x, target_y 
    
    
    last_x, last_y = position_history[-1]
    prev_x, prev_y = position_history[-2]
    vx = (last_x - prev_x) / frame_time  
    vy = (last_y - prev_y) / frame_time  
    
    
    predicted_x = target_x + vx * frame_time
    predicted_y = target_y + vy * frame_time
    return predicted_x, predicted_y


def move_mouse_to_target(target_x, target_y, position_history=None):
    current_x, current_y = win32api.GetCursorPos()
    frame_time = 1.0 / settings["performance"]["target_fps"]["value"]
    
    
    if settings["aiming"]["movement_prediction"]["enabled"]["value"] and position_history:
        target_x, target_y = predict_next_position(target_x, target_y, position_history, frame_time)
        if settings["debug"]["enable_debug"]["value"]:
            print(f"Predicted position: ({target_x}, {target_y})")

    dx_total = target_x - current_x
    dy_total = target_y - current_y
    steps = settings["aiming"]["interpolation_steps"]["value"]

    for i in range(steps):
        current_x, current_y = win32api.GetCursorPos()  
        dx_remaining = target_x - current_x
        dy_remaining = target_y - current_y
        
        dx = int(dx_remaining * settings["aiming"]["smooth_factor"]["value"] + 
                 random.uniform(settings["aiming"]["jitter_min"]["value"], settings["aiming"]["jitter_max"]["value"]))
        dy = int(dy_remaining * settings["aiming"]["smooth_factor"]["value"] + 
                 random.uniform(settings["aiming"]["jitter_min"]["value"], settings["aiming"]["jitter_max"]["value"]))
        
        dx = max(min(dx, settings["aiming"]["max_move_speed"]["value"]), -settings["aiming"]["max_move_speed"]["value"])
        dy = max(min(dy, settings["aiming"]["max_move_speed"]["value"]), -settings["aiming"]["max_move_speed"]["value"])
        win32api.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
        delay = random.uniform(settings["aiming"]["movement_delay_min"]["value"], 
                               settings["aiming"]["movement_delay_max"]["value"])
        time.sleep(delay)
        if settings["debug"]["enable_debug"]["value"]:
            new_x, new_y = win32api.GetCursorPos()
            print(f"Mouse step {i}/{steps-1}: Dx={dx}, Dy={dy}, New Pos: ({new_x}, {new_y}), Delay: {delay:.4f}s")


def aim_at_target(targets, screen_width, screen_height, position_history):
    if targets:
        closest_target = min(targets, key=lambda t: (t["center"][0] - settings["detection"]["detection_radius"]["value"]) ** 2 + 
                                                    (t["center"][1] - settings["detection"]["detection_radius"]["value"]) ** 2)
        target_x = closest_target["screen_x"]
        target_y = closest_target["screen_y"]

        current_x, current_y = win32api.GetCursorPos()
        if settings["debug"]["enable_debug"]["value"]:
            print(f"Aiming at: ({target_x}, {target_y}), Current: ({current_x}, {current_y})")
        
        
        position_history.append((target_x, target_y))
        if len(position_history) > settings["aiming"]["movement_prediction"]["prediction_frames"]["value"]:
            position_history.popleft()
        
        move_mouse_to_target(target_x, target_y, position_history)
    elif settings["debug"]["enable_debug"]["value"]:
        print("No targets to aim at")


def create_overlay():
    pygame.init()
    screen_info = pygame.display.Info()
    screen_width, screen_height = screen_info.current_w, screen_info.current_h

    overlay_size = settings["detection"]["detection_radius"]["value"] * 2
    center_x, center_y = screen_width // 2, screen_height // 2

    screen = pygame.display.set_mode((overlay_size, overlay_size), pygame.NOFRAME)
    pygame.display.set_caption("Aimbot Overlay")

    hwnd = pygame.display.get_wm_info()['window']
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, center_x - settings["detection"]["detection_radius"]["value"], 
                          center_y - settings["detection"]["detection_radius"]["value"], 0, 0, win32con.SWP_NOSIZE)

    styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    styles |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)

    transparent_color = (0, 0, 0)
    win32gui.SetLayeredWindowAttributes(hwnd, 0, int(255 * settings["overlay"]["overlay_alpha"]["value"]), win32con.LWA_COLORKEY | win32con.LWA_ALPHA)

    font = pygame.font.SysFont(None, settings["overlay"]["font_size"]["value"])

    return screen, transparent_color, screen_width, screen_height, font

# Main loop
def main():
    overlay, transparent_color, SCREEN_WIDTH, SCREEN_HEIGHT, font = create_overlay()
    print(f"Aimbot ready. Press '{settings['controls']['toggle_aimbot_key']['value']}' to toggle aimbot. "
          f"Press '{settings['controls']['toggle_aim_only_key']['value']}' to toggle Aim Only mode. Press Ctrl+C to stop.")
    running = True
    clock = pygame.time.Clock()
    frame_count = 0
    latest_targets = []  
    position_history = deque(maxlen=settings["aiming"]["movement_prediction"]["prediction_frames"]["value"])  # Track target positions

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            overlay.fill(transparent_color)

            should_aim = aimbot_enabled and (not aim_only_enabled or right_click_held)
            if settings["debug"]["enable_debug"]["value"]:
                print(f"Should aim: {should_aim}")

            if should_aim:
                pygame.draw.circle(overlay, tuple(settings["overlay"]["circle_color"]["value"]), 
                                  (settings["detection"]["detection_radius"]["value"], settings["detection"]["detection_radius"]["value"]), 
                                  settings["detection"]["detection_radius"]["value"], settings["overlay"]["circle_width"]["value"])
               
                if frame_count % settings["performance"]["detection_interval"]["value"] == 0:
                    latest_targets = detect_players(SCREEN_WIDTH, SCREEN_HEIGHT)
                
                aim_at_target(latest_targets, SCREEN_WIDTH, SCREEN_HEIGHT, position_history)

                center = (settings["detection"]["detection_radius"]["value"], settings["detection"]["detection_radius"]["value"])
                for target in latest_targets:
                    pygame.draw.line(overlay, tuple(settings["overlay"]["line_color"]["value"]), center, target["center"], 
                                    settings["overlay"]["line_width"]["value"])
                    x_min, y_min, x_max, y_max = [int(v) for v in target["box"]]
                    pygame.draw.rect(overlay, tuple(settings["overlay"]["box_color"]["value"]), 
                                    (x_min, y_min, x_max - x_min, y_max - y_min), settings["overlay"]["line_width"]["value"])
                    conf_text = font.render(f"{target['conf']:.2f}", True, tuple(settings["overlay"]["text_color"]["value"]))
                    overlay.blit(conf_text, (x_min, y_min - 20 if y_min - 20 > 0 else y_min + 5))

            pygame.display.flip()
            clock.tick(settings["performance"]["target_fps"]["value"])
            frame_count += 1

    except KeyboardInterrupt:
        print("Aimbot stopped.")
    finally:
        listener.stop()
        pygame.quit()

if __name__ == "__main__":
    main()