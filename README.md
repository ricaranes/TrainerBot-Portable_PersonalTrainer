# Trainer Bot - Portable Personal Trainer

## Trainer Bot
<img align="center" width="480" height="250" alt="GIF" src="https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/a90799fb-6133-44ea-b37d-33a6ca0e04b9.gif" />

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

## Project Description
As a former lover of sports sciences, I believe that the process of becoming the happiest version of ourselves starts from the inside out, as this becomes fully connected with improving our mental and physical health, whether it is at home, at the gym, or from a distance. Workout training is known as a form to combine this cyclic web of well-being. Specially with the previous pandemic and lockdown that changed our world, the online game became really strong when it comes to online training and another topics of interest in our lives. When I thought about adding value to future experiences in online training, I decided to create something useful, accessible, and enjoyable for everyone. So I did it, and Trainer Bot is finally here! Trainer Bot is an application that acts as a portable personal trainer, providing real-time feedback to improve the execution of "Squat" and "Deadlift" physical exercises. This dinamic app utilizes computer vision techniques to analyze the user's movements while performing the exercises and offers immediate feedback to correct possible execution errors. Finally, we have an ally to keep our body moving without excuses!

## How It Works
Trainer Bot was developed using the Python programming language and the Flask library to create an interactive web interface. The application uses Google's MediaPipe library, which provides an API for real-time detection and tracking of body landmarks from images or videos. These landmarks are key points that represent joints and parts of human body (from head to toes, so nothing is missing). 

The app offers two exercise options: "Squat" and "Deadlift." The user can select the desired exercise using keyboard commands:

"1" to select "Squat"
"2" to select "Deadlift"
"f" or "F" to view the exercises from the front view
"s" or "S" to view the exercises from the side view
"e" or "E" to end the exercise
"r" or "R" to restart the repetition count
Trainer Bot uses the detected body landmarks to analyze the execution of exercises in real-time and provide immediate feedback to the user. The feedback is given through text displayed on the video and audio feedback, utilizing the Text-to-Speech (TTS) module of the operating system.

The application also features a repetition counting functionality, where it monitors the movement of knees and wrists to determine a complete repetition. For each well-executed repetition, Trainer Bot increments the count of successful repetitions. It is really as rigorous as it seems!

## What it looks like
## Initial Page
![photo](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/initial%20page.png)

## Main Page
![photo](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/Main%20page.png)

## Resume Exercice Page
![photo](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/score%20page.png)

## Results Page
![photo](https://github.com/ricaranes/TrainerBot-Portable_PersonalTrainer/blob/main/git%20images/results%20page.png)

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
