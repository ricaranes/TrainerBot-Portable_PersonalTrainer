# Trainer Bot - Portable Personal Trainer

## Trainer Bot
[![video]([caminho/para/o/seu/video.mp4](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/logo.mp4))]([caminho/para/o/seu/video.mp4](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/logo.mp4))

## Project Description
Trainer Bot is an application that acts as a portable personal trainer, providing real-time feedback to improve the execution of "Squat" and "Deadlift" physical exercises. The app utilizes computer vision techniques to analyze the user's movements while performing the exercises and offers immediate feedback to correct possible execution errors.

## How It Works
Trainer Bot was developed using the Python programming language and the Flask library to create an interactive web interface. The application uses Google's MediaPipe library, which provides an API for real-time detection and tracking of body landmarks from images or videos. These landmarks are key points that represent joints and parts of the human body.

The app offers two exercise options: "Squat" and "Deadlift." The user can select the desired exercise using keyboard commands:

"1" to select "Squat"
"2" to select "Deadlift"
"f" or "F" to view the exercises from the front view
"s" or "S" to view the exercises from the side view
"e" or "E" to end the exercise
"r" or "R" to restart the repetition count
Trainer Bot uses the detected body landmarks to analyze the execution of exercises in real-time and provide immediate feedback to the user. The feedback is given through text displayed on the video and audio feedback, utilizing the Text-to-Speech (TTS) module of the operating system.

The application also features a repetition counting functionality, where it monitors the movement of knees and wrists to determine a complete repetition. For each well-executed repetition, Trainer Bot increments the count of successful repetitions.

## What it looks like
## Initial Page


## Main Page


## Resume Exercice Page


## Results Page

## How to Use
Make sure you have Python installed on your system.
Install the required libraries using the command pip install flask mediapipe pytesseract opencv-python.
Run the Trainer Bot code using Python.
Access the Trainer Bot's web interface through your browser at http://localhost:5000.
Select the desired exercise by typing "1" for "Squat" or "2" for "Deadlift."
Choose the desired view by typing "f" or "F" for the front view or "s" or "S" for the side view.
Perform the exercises in front of the camera, and Trainer Bot will provide real-time feedback on the execution of movements.
When you finish the exercise, press "e" or "E" to end it.
Trainer Bot will display a summary of the results and final feedback on performance.
## Resources and Technologies Used
Python
Flask (Web Framework)
MediaPipe (Computer Vision Library)
OpenCV (Library for image and video processing)
pytesseract (OCR Library for text detection in images)
