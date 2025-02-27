# Nth-Overkill
A free, open-source external universal aimbot developed and maintained in python.

# How It Works
Nth-Overkill is an Simple external aimbot designed for FPS games, built using computer vision and low-level Windows API calls. Leveraging the YOLOv5 object detection model (yolov5n), it captures the screen in real-time, detects players within a configurable radius, and moves the mouse to aim at them (which is what an aimbot does).

# How To Use/Install
To use this program, you must have vscode or some other code editor, python 3.12.4 or above, the python extension (optionial, vscode), and the nessecary packages (and make sure you're in the right directory). Download and extract the Nth-Overkill folder. You can install all the required packages with this command (in the code editor terminal or cmd): 
```bash
pip install opencv-python torch numpy pygame keyboard pywin32 mss pynput
```
Drag the extracted Nth-Overkill folder inside your workspace. Once you do that, you are pretty much good to go! Just run the script and a options.json file should appear in the folder with the script (if it is not there already). There are many options in there with descriptions, although make sure to restart the program when you change a setting.

# Suggestions/Help
1. Make sure to use smaller guns with unobstructive sights for best results, as the YOLOv5 model works best when it can see the full body, on a similar note, make sure the people/creatures you're fighing are vaguely humanoid.
2. The movment prediction setting is in very early stages, so i would not recommend it for regular gameplay.
3. let me know about any bugs or complaints on this Repo.
4. The best way to get results is to train the model for the game you're playing, but if you don't know how to do that, then you can also tweak the settings to work for the game you're currently playing. The universal settings will work for most 
   applications, but it works best when you customize it for the game you're using it on.

# Controls
```Num1 to toggle aimbot```
```Num2 to toggle aim-only mode```

# Notes
This aimbot is still very early in the development process, so please forgive the bugs/jank you may encounter.

# Anyways, thats pretty much it! (for now)
