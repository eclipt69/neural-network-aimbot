import numpy as np
from termcolor import colored
import timeit
import _thread
import imutils
import time
import mss
import cv2
import os
import signal
import sys
import pynput
import ctypes
import keyboard
import win32gui, win32ui, win32con, win32api
from colorama import Fore

##########################################################################
# Neural Network Aimbot Using YOLO and Cv2                               #
# Made by Eclipt#8243 as a Majoris side project!                         #
# Make sure to subscribe me @ https://www.youtube.com/@eclipt_2728       #
# And join my discord :-)                                                #
##########################################################################



WARN = f"[{Fore.RED}WARN{Fore.RESET}]" #WARNING
INFO = f"[{Fore.CYAN}INFO{Fore.RESET}]" #INFORMATION
STAT = f"[{Fore.YELLOW}STAT{Fore.RESET}]" #STATUS
#BMOD -> BOT MODE

version = "2.5.0"

sct = mss.mss()
Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def set_pos(x, y):
    x = 1 + int(x * 65536./Wd)
    y = 1 + int(y * 65536./Hd)
    extra = ctypes.c_ulong(0)
    ii_ = pynput._util.win32.INPUT_union()
    ii_.mi = pynput._util.win32.MOUSEINPUT(x, y, 0, (0x0001 | 0x8000), 0, ctypes.cast(ctypes.pointer(extra), ctypes.c_void_p))
    command=pynput._util.win32.INPUT(ctypes.c_ulong(0), ii_)
    SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
if __name__ == "__main__":
    from colorama import Fore
    print(f"[WARNING] You are booting on the wrong directory!")

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

checkBlocker = False
def start(onEnable):

    # Configs
    yoloModelDir = "models"
    botConfidence = 0.36
    threshold = 0.22

    activeRange = 400

    labelsPath = os.path.sep.join([yoloModelDir, "coco-dataset.labels"])
    LABELS = open(labelsPath).read().strip().split("\n")

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    weightsPath = os.path.sep.join([yoloModelDir, "yolov3-tiny.weights"])
    configPath = os.path.sep.join([yoloModelDir, "yolov3-tiny.cfg"])


    time.sleep(0.4)

    print(f"{INFO} Loading Neural Network Models...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Screen capture
    print(f"{INFO} Loading Screencapture Window...")
    print(f"{INFO} Loading Screencapture Data Collector...")
    W, H = None, None
    origbox = (int(Wd/2 - activeRange/2),
               int(Hd/2 - activeRange/2),
               int(Wd/2 + activeRange/2),
               int(Hd/2 + activeRange/2))

    # Loggings
    if not onEnable:
        print(f"{INFO} Adaii Aimbot is [{Fore.RED}Offline{Fore.RESET}]")
        print(f"{Fore.RED}{STAT} Using Visualizer-Only Mode!{Fore.RESET}")
    else:
        print(f"{INFO} Adaii Aimbot is [{Fore.GREEN}Online{Fore.RESET}]")

    # Handle Ctrl+C & Release pointers
    def signal_handler(sig, frame):
        # release files
        print()
        print(f"\n[{Fore.YELLOW}BMOD{Fore.RESET}] {Fore.YELLOW}Finnishing Session, please wait...{Fore.RESET}")
        print(f"{STAT} Shutting Down Screencapture...")
        print(f"{STAT} Shutting Down AdaiiBot...")
        time.sleep(2)
        print(f"{INFO} Session Finished Successfully")
        time.sleep(1)
        sct.close()
        cv2.destroyAllWindows()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
        
    # GPU Support
    build_info = str("".join(cv2.getBuildInformation().split()))
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(False)
        cv2.ocl.useOpenCL()
        print(f"{STAT} OpenCL Acceleration [{Fore.GREEN}Online{Fore.RESET}]")
    else:
        print(f"{WARN} OpenCL Acceleration [{Fore.RED}Offline{Fore.RESET}]")
    if "CUDA:YES" in build_info:
        print(f"{STAT} CUDA Acceleration [{Fore.GREEN}Online{Fore.RESET}]")
    else:
        print(f"{WARN} CUDA Acceleration [{Fore.RED}Offline{Fore.RESET}]")
    print(f"\n{INFO} Use ctrl+c or q to quit.")

    botStatus = "Online"

    def bot_stats():
        if abot==2:
            aimbot_statusss = colored("ONLINE", 'green')
        elif abot==1:
            aimbot_statusss = colored("OFFLINE", 'red')
        else:
            aimbot_statusss = colored("OFFLINE", 'red')
            
        sys.stdout.write("\033[K")
        print(f"{STAT} Aimbot [{aimbot_statusss}]! (P)", end = "\r")
    print()


    while True:
        start_time = timeit.default_timer()
        frame = np.array(grab_screen(region=origbox))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[: 2]

        frame = cv2.UMat(frame)

        blob = cv2.dnn.blobFromImage(frame, 1 / 260, (150, 150),
                                     swapRB=False, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)


        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            
            for detection in output:
                scores = detection[5:]

                # classID = np.argmax(scores)
                # confidence = scores[classID]
                classID = 0  # person = 0
                confidence = scores[classID]

                # filter weak predictions
                if confidence > botConfidence:
                    box = detection[0: 4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y) and box coords
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update bounding box coordinates,
                    # confidences, and class IDs lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, botConfidence, threshold)

        if len(idxs) > 0:

            # Best match
            bestMatch = confidences[np.argmax(confidences)]

            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Drawings
                cv2.circle(frame, (int(x + w / 2), int(y + h / 5)), 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (x, y),
                                (x + w, y + h), (0, 0, 255), 2)
                
                text = "TARGETING {}%".format(int(confidences[i] * 100))
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if onEnable and bestMatch == confidences[i]:
                    mouseX = origbox[0] + (x + w/1.5)
                    mouseY = origbox[1] + (y + h/5)
                    set_pos(mouseX, mouseY)
        
        elapsed = timeit.default_timer() - start_time
        Status = " {1} FPS / {0} MS Interpol* Delay \t".format(int(elapsed*1000), int(1/elapsed))
        eyeStatus = " {1} FPS & {0} MS \t".format(int(elapsed*1000), int(1/elapsed))
        cv2.imshow(f"eye", frame)
        cv2.setWindowTitle(f'eye', f'{botStatus} |{eyeStatus}')

        sys.stdout.write(
            "\r" + f"[{Fore.RED}♦{Fore.RESET}] {INFO}" + " {1} FPS & {0} MS Interpol* Delay \t".format(int(elapsed*1000), int(1/elapsed)))
        os.system(f"title Running Adaii v{version}  {botStatus} {Status} ~by Eclipt#8243")
        sys.stdout.flush()

        if not checkBlocker:
            if onEnable==True:
                abot = 2
                bot_stats()
            elif onEnable==False:
                abot = 1
                bot_stats()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if keyboard.is_pressed('p') and onEnable==True:
            onEnable=False
            abot = 1
            bot_stats()
            botStatus = "Offline"
            time.sleep(0.1)

        elif keyboard.is_pressed('p') and onEnable==False:
            onEnable=True
            abot = 2
            bot_stats()
            botStatus = "Online"
            time.sleep(0.1)
        elif keyboard.is_pressed('q'):
        	break
            

    # Clean up on exit
    signal_handler(0, 0)
