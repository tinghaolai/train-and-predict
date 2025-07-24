import cv2
import numpy as np
import mss
import win32gui
import win32con
import pyautogui
import time
import keyboard

minimap_template = cv2.imread('minimap_template.png')
player_icon = cv2.imread('player_icon.png')

# 儲存圖片看是要擷取下面的 sct.monitors[X] index 多少
# with mss.mss() as sct:
#     for i, monitor in enumerate(sct.monitors):
#         print(f"📸 Capturing Monitor {i}: {monitor}")
#         screenshot = sct.grab(monitor)
#         img = np.array(screenshot)
#
#         filename = f"monitor_{i}.png"
#         cv2.imwrite(filename, img)
#         print(f"✅ Saved {filename}")

with mss.mss() as sct:
    for i, monitor in enumerate(sct.monitors):
        print(f"Monitor {i}: {monitor}")
    monitor = sct.monitors[3]
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

# search mini map
# result = cv2.matchTemplate(screenshot, minimap_template, cv2.TM_CCOEFF_NORMED)
# minimap_threshold = 0.8
# minimap_locations = np.where(result >= minimap_threshold)

# if len(minimap_locations[0]) == 0:
#     print("❌ mini map not found")
#     exit()
#
# fount location of mini map
# y, x = minimap_locations[0][0], minimap_locations[1][0]
# minimap_h, minimap_w = minimap_template.shape[:2]
# minimap_roi = screenshot[y:y+minimap_h, x:x+minimap_w]

# result2 = cv2.matchTemplate(minimap_roi, player_icon, cv2.TM_CCOEFF_NORMED)
# player_threshold = 0.8
# player_locations = np.where(result2 >= player_threshold)

# if len(player_locations[0]) == 0:
#     print("❌ character not found")
#     exit()


WINDOW_TITLE = 'Your windows name'

def bring_window_to_front(title):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    return False

if bring_window_to_front(WINDOW_TITLE):
    time.sleep(0.5)

    start_time = time.time()

    pyautogui.keyDown('right')
    while time.time() - start_time < 10:
        keyboard.press_and_release('s')
        time.sleep(0.1)

        # pyautogui.keyUp('s')

        # keyboard.press_and_release('right')
        # time.sleep(0.05)
        #
        keyboard.press_and_release('c')
        time.sleep(0.1)
        # pyautogui.keyUp('right')

    # pyautogui.keyUp('right')
    print("✅ cycle finished")
else:
    print("❌ window not founded：", WINDOW_TITLE)