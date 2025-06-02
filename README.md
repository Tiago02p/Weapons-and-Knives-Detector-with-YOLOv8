# Weapon and Knievs Detection System

This repository contains the source code and resources related to the academic weapon detection project, developed as part of research in the area of ​​computer security.

## Overview

The main objective of this project is to design and implement an advanced system for the autonomous detection of firearms and knives. Using the YOLOv8 (You Only Look Once) framework and transfer learning techniques, we seek to improve security effectiveness through continuous, real-time surveillance.

## Key Features

- **YOLOv8 Framework:** One implementation uses YOLOv8, known for its efficiency in real-time object detection.
  
- **Transfer Learning:** Transfer learning techniques are employed to adapt the model to a specific context and improve accuracy in weapon detection.

- **Integration with IP Cameras:** The system is designed for easy integration with IP cameras, allowing for real-time surveillance and immediate notifications.

- **Real-time Webcam Detection:** The system includes real-time weapon detection using your computer's webcam, featuring:
  - Live object tracking with BoTSORT
  - FPS monitoring
  - Real-time visualization
  - Efficient CPU usage
  - Error handling and stability improvements

## Dataset

https://universe.roboflow.com/joao-assalim-xmovq/weapon-2/dataset/2

## How to use

1. **Repository Cloning:**

```
git clone https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8.git
```

2. **Installation of dependencies:**

```
pip install -r requirements.txt
```

3. **System Execution:**

For image detection:
```
python detecting-images.py
```

For real-time webcam detection:
```
python real_time_detection.py
```

The real-time detection system will:
- Open your webcam
- Start detecting weapons in real-time
- Track detected objects between frames
- Display FPS counter
- Press 'q' to quit the program

## Contributions and Problems

Contributions are welcome! If you encounter issues or have suggestions for improvement, please open an issue in this repository.

## Academic Notes

This project is part of academic research in the area of ​​computer security. The results obtained and performance analyzes are documented in detail in the scientific article that will be made available in the future.

## License

This project is distributed under the [MIT] license (LICENSE.md). See the LICENSE.md file for details.
