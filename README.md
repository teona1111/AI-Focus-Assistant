# AI Focus Assistant

## Overview

AI Focus Assistant is a system that monitors a user's focus level using two independent sources of data:

- Webcam (CNN model for eye state detection)
- STM32 accelerometer and gyroscope (Random Forest model)

The predictions are combined by a Flask server to determine whether the user is focused or distracted and are displayed on a live dashboard.

---

## Features

- Eye state detection using a CNN model
- Motion analysis using a Random Forest model
- Sensor data acquisition from STM32
- Prediction fusion
- Live web dashboard
- Real-time monitoring

---

## Technologies

- Python
- Flask
- TensorFlow (Keras API)
- Scikit-learn
- OpenCV
- MediaPipe
- STM32
- HTML/CSS/JavaScript

---

## Requirements

Before running the project, ensure you have:

- Python 3.11
- A webcam
- STM32 board
- All required Python packages listed in `requirements.txt`

---

## Quick Start

1. Clone the repository.
2. Install dependencies.
3. Run the application.
4. Open the dashboard in your browser.

Detailed installation instructions are available in the GitLab Wiki.

---

## Documentation

Complete project documentation is available in the GitLab Wiki.

The Wiki contains:

- Project Specifications
- Installation Guide
- Use Cases
- Feature Documentation

---

## Authors

- Teona Mehandjiska
- Anastasija Lukarova
- Filip Tanevski
