from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import time
import subprocess
import pytesseract
from flask import redirect
from flask import current_app
import os
os.environ['PATH'] += ':/usr/local/bin'

from functions import text_to_speech, exibir_contagem, analisar_frame
from processar_exercicios import processar_squat_frontal, processar_squat_lateral, processar_deadlift_frontal, processar_deadlift_lateral

app = Flask(__name__, template_folder='/Users/macbook/Documents/Trainer BOT/Templates')
app.secret_key = 'chave_secreta_aqui'



repetitions = 0
good_repetitions = 0
contagem_ativa = False
frame_sem_erros = True

import numpy as np

#Function responsible for analyzing each repetition's 
# frame and distinguishing between well-executed and poorly executed repetitions.

def analisar_frame(frame):
    # Define the range of red color.
    vermelho_min = np.array([0, 0, 200], dtype=np.uint8)
    vermelho_max = np.array([50, 50, 255], dtype=np.uint8)

    # Create a mask to segment the red color in the image.
    mask = cv2.inRange(frame, vermelho_min, vermelho_max)

    # Apply the mask to extract only the red pixels.
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert the resulting image to grayscale.
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Perform text detection on the grayscale image using Optical Character Recognition (OCR).
    text = pytesseract.image_to_string(gray)

    # Check if the error message is present in the text.
    if "Error:" in text:
        return False  # Repetition with errors
    else:
        return True  # Well-executed repetition
    

 
 #Error detection logic for the front-view squat exercise   
def processar_squat_frontal(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros, key

    # Checks if results are available
    if results is None or not results.pose_landmarks:
        return frame_bgr
    
    
    # Logic for the front plane
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
    
        # Error check: knees inward
        left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
        left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
        right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x

        if left_knee_x < left_ankle_x or right_knee_x > right_ankle_x:
            cv2.putText(frame_bgr, "Error: Knees IN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame_sem_erros = False

        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Error check: Hips above Knees
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
        left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y

        if left_knee_y > left_hip or right_knee_y > right_hip:
            cv2.putText(frame_bgr, "Error: Hips above Knees ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame_sem_erros = False

        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # repetition count    
        if left_knee_y < left_hip or right_knee_y < right_hip:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Save the frame to a file
                cv2.imwrite("frame.jpg", frame_bgr)

                # 1 millisecond pause after each repetition
                time.sleep(0.1)

                # Perform the analysis of the recorded frame
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Check that the recorded replay was well done
                if repeticao_bem_feita:
                    # Increase the count of well-executed repetitions
                    good_repetitions += 1
                   
                    print("Repetição bem executada!")
                else:
                   
                    print("Repetição com erros!")

        else:
            contagem_ativa = False

    return frame_bgr


# Function to process the frontal plane of the "deadlift" exercise
def processar_deadlift_frontal(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros, key
    # Check if there are results available
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Logic for the front view
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
       
         # Shoulder lift Error
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        mouth_right = results.pose_landmarks.landmark[9]
        mouth_left = results.pose_landmarks.landmark[10]

        shoulder_to_mouth_distance = abs(shoulder_right.y - mouth_right.y)  # Vertical distance between right shoulder and right mouth
        shoulder_to_mouth_distance += abs(shoulder_left.y - mouth_left.y)  # Add vertical distance between left shoulder and left mouth
        shoulder_to_mouth_distance /= 2  # Calculate the average of the distances

        if shoulder_to_mouth_distance < 0.08:  # Reference value for shoulder elevation detection
            cv2.putText(frame_bgr, "Error: High Shoulders", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Error: Knees in
        left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
        left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
        right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
        
        threshold = 0.02

        if left_knee_x < left_ankle_x - threshold or right_knee_x > right_ankle_x + threshold:
            cv2.putText(frame_bgr, "Error: Knees IN", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # repetition count
        # Get coordinates of wrist and knee landmarks
        right_wrist_y = results.pose_landmarks.landmark[16].y  # right wrist
        left_wrist_y = results.pose_landmarks.landmark[15].y  # left wrist
        right_knee_y = results.pose_landmarks.landmark[26].y  # Right knee
        left_knee_y = results.pose_landmarks.landmark[25].y  # Left knee

        # Checks if the wrist landmarks are at or below the knees
        if right_wrist_y <= right_knee_y or left_wrist_y <= left_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Save the frame to a file
                cv2.imwrite("frame.jpg", frame_bgr)

                # 0.1 second pause after each repetition
                time.sleep(0.1)

                # Perform the analysis of the recorded frame
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Check that the recorded replay was well done
                if repeticao_bem_feita:
                    # Increase the count of well-executed repetitions
                    good_repetitions += 1
                    # print positive feedback
                    print("Repetição bem executada!")
                else:
                    # print negative feedback
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
            
    return frame_bgr



# Function to process the side view of the squat
def processar_squat_lateral(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros
    # Checks if results are available
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Logic for the side view
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
        # Coordinates for error detection in the side view
        left_knee = results.pose_landmarks.landmark[25].x
        right_knee = results.pose_landmarks.landmark[26].x
        left_ankle = results.pose_landmarks.landmark[31].x
        right_ankle = results.pose_landmarks.landmark[32].x
        left_shoulder = results.pose_landmarks.landmark[11].x
        right_shoulder = results.pose_landmarks.landmark[12].x

        #Error check: Knees passing the toes
        threshold_distance_knee = 0.03
        if left_knee + threshold_distance_knee < left_ankle or right_knee + threshold_distance_knee < right_ankle:
            cv2.putText(frame_bgr, "Error: Knees Too Far Forward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #Error check: Shoulders passing the knees
        threshold_distance_shoulder = 0.03
        if left_shoulder + threshold_distance_shoulder < left_knee or right_shoulder + threshold_distance_shoulder < right_knee:
            cv2.putText(frame_bgr, "Error: Trunk Leaning Too Far Forward", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # repetition count
        left_hip_y = results.pose_landmarks.landmark[23].y  
        right_hip_y = results.pose_landmarks.landmark[24].y  
        left_knee_y = results.pose_landmarks.landmark[25].y  
        right_knee_y = results.pose_landmarks.landmark[26].y  

            

        # Check that the hip is below the knee
        threshold_distance_hip = 0.1
        if left_hip_y + threshold_distance_hip <= left_knee_y or right_hip_y + threshold_distance_hip <= right_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Save the frame to a file
                cv2.imwrite("frame.jpg", frame_bgr)

                # 0.1 second pause after each repetition
                time.sleep(0.1)

                # Perform the analysis of the recorded frame
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Check that the recorded replay was well done
                if repeticao_bem_feita:
                    # Increase the count of well-executed repetitions
                    good_repetitions += 1
                    # Print positive feedback
                    print("Repetição bem executada!")
                else:
                    # print negative feedback
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
        
    return frame_bgr


# Function to process the lateral plane of the "deadlift" exercise
def processar_deadlift_lateral(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros
    # Checks if results are available
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Logic for the side plane
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
        # arched back
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        shoulder_to_mouth_distance = abs(shoulder_right.y - mouth_right.y)  # Vertical distance between right shoulder and right mouth
        shoulder_to_mouth_distance += abs(shoulder_left.y - mouth_left.y)  # Adds vertical distance between left shoulder and left mouth
        shoulder_to_mouth_distance /= 2  # Calculates the average of the distances

        if shoulder_to_mouth_distance < 0.08 and nose.y > shoulder_right.y and nose.y > shoulder_left.y:
            cv2.putText(frame_bgr, "Error: Arched Back", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # knee and hip coordinates

        left_knee = results.pose_landmarks.landmark[25].x
        right_knee = results.pose_landmarks.landmark[26].x
        left_ankle = results.pose_landmarks.landmark[31].x
        right_ankle = results.pose_landmarks.landmark[32].x

        knee_threshold = 0.01  # Difference limit between knees and toes

        # Error check: Knees Passing toes
        if left_knee < left_ankle - knee_threshold or right_knee < right_ankle - knee_threshold:
            cv2.putText(frame_bgr, "Error: Knees Too Far Forward", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Excessive trunk forward lean
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        knee_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

        shoulder_knee_threshold = 0.06  # Horizontal difference threshold between shoulders and knees

        if abs(shoulder_right.x - knee_right.x) > shoulder_knee_threshold or abs(shoulder_left.x - knee_left.x) > shoulder_knee_threshold:
            cv2.putText(frame_bgr, "Error: Excessive Trunk Forward Lean", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # repetition count
        # Get coordinates of wrist and knee landmarks
        right_wrist_y = results.pose_landmarks.landmark[16].y  # right wrist
        left_wrist_y = results.pose_landmarks.landmark[15].y  # left wrist
        right_knee_y = results.pose_landmarks.landmark[26].y  # Right knee
        left_knee_y = results.pose_landmarks.landmark[25].y  # Left knee

        # Checks if the wrist landmarks are at or below the knees
        if right_wrist_y <= right_knee_y or left_wrist_y <= left_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Save the frame to a file
                cv2.imwrite("frame.jpg", frame_bgr)

                # 0.1 second pause after each repetition
                time.sleep(0.1)

                # Perform the analysis of the recorded frame
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Check that the recorded replay was well done
                if repeticao_bem_feita:
                    # Increase the count of well-executed repetitions
                    good_repetitions += 1
                    # Print positive feedback
                    print("Repetição bem executada!")
                else:
                    # print negative feedback
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
            

    return frame_bgr


#text to speech function
def text_to_speech(text):
    # Time check for audio feedback
    current_time = time.time()
    if current_time - text_to_speech.last_feedback_time >= text_to_speech.feedback_interval:
        text_to_speech.last_feedback_time = current_time
        subprocess.call(["say", text])
#function to display permanent count on video        
def exibir_contagem(frame_bgr):
    global repetitions
    cv2.putText(frame_bgr, f"Repetitions: {repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame_bgr

# Variables for controlling time
text_to_speech.last_feedback_time = time.time()
text_to_speech.feedback_interval = 10  # Time interval between feedbacks in seconds







# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Exercise and view control variables
exercicio = None
plano_frontal = False
plano_lateral = False
repetitions = 0
good_repetitions = 0
key = 0  # Set key initially
exercise_ended = False
exercise_name = ''  # Variable to store the exercise name


# Variables to control the time
last_feedback_time = time.time()
feedback_interval = 5  # Time interval between feedbacks in seconds

# Input video setup
cap = cv2.VideoCapture(0)  # Use 0 for the default system camera

# MediaPipe Pose initialization
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=True,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

# Route to display the welcome page
    @app.route('/')
    def welcome():
        return render_template('welcome.html')

# Route to display the index page
    @app.route('/index')
    def index():
        return render_template('index.html')

   
# Route to handle the keypress event sent via POST request
    @app.route('/keypress', methods=['POST'])
    def keypress():
        global exercicio, plano_frontal, plano_lateral, key, repetitions

        key = request.form['key']
        # Check which key was pressed and update the global variables accordingly
        if key == '1':
            exercicio = 'squat'
            exercise_name = 'Squat'
            print("Exercise: Squat")
        elif key == '2':
            exercicio = 'deadlift'
            exercise_name = 'Deadlift'
            print("Exercise: Deadlift")
        elif key == 'f' or key == 'F':
            plano_frontal = True
            plano_lateral = False
            print("Plano: Frontal")
        elif key == 's' or key == 'S':
            plano_frontal = False
            plano_lateral = True
            print("Plano: Lateral")
        elif key == 'e' or key == 'E':
            exercise_ended = True
        elif key == 'r' or key == 'R':
            repetitions = 0
            
            print("Repetitions reset")

        return 'OK'
    
    # Route to end the exercise and redirect to the 'well_done' page
    @app.route('/end_exercise')
    def end_exercise():
        global exercise_ended
        exercise_ended = True
        return redirect('/well_done')
    
    # Route to display the 'well_done' page
    @app.route('/well_done')
    def well_done():
        global exercise_name, repetitions, good_repetitions
        return render_template('well_done.html', repetitions=repetitions, exercise_name=exercicio, good_repetitions=good_repetitions)
    
   
    # Route to display the 'results' page
    @app.route('/results')
    def results():
        global exercise_name, repetitions, good_repetitions
        # Get the values of repetitions, good_repetitions, and exercise_name from the query parameters
        repetitions = int(request.args.get('repetitions'))
        good_repetitions = int(request.args.get('good_repetitions'))
        exercise_name = request.args.get('exercise_name')

        # Based on the exercise type, generate a result message
        if exercicio == 'squat':
            if good_repetitions >= repetitions / 2:
                result_message = "Great job! Nothing To Add"
            else:
                result_message = "You need to improve your squat. Let's work on enhancing your technique:\n\n"
                result_message += "Here are some tips:\n\n"
                result_message += "- Set your feet at shoulder-width apart with toes slightly pointing outward.\n\n"
                result_message += "- Keep your chest lifted and your back straight throughout the movement.\n\n"
                result_message += "Keep practicing, and your squat will become even more solid!"


        elif exercicio == 'deadlift':
            if good_repetitions >= repetitions / 2:
                result_message = "Great job! Nothing To Add"
            else:
                result_message = "You need to improve your deadlift. Let's focus on these tips:\n\n"
                result_message += "Here are some tips:\n\n"
                result_message += "- Ensure your starting position is correct – feet shoulder-width apart, toes slightly turned out, and a neutral spine.\n\n"
                result_message += "- Warm-up properly and engage your core to protect your back during heavy lifts.\n\n"
                result_message += "Believe in yourself, keep practicing, and soon, you'll conquer the deadlift like a pro!"


        # Call the Text-to-Speech function to read the message
        text_to_speech(result_message)


        return render_template('results.html', repetitions=repetitions, good_repetitions=good_repetitions, exercise_name=exercicio, result_message=result_message)

        
    

    # Route for video streaming
    def generate_frames():
        global exercicio, plano_frontal, plano_lateral, repetitions, good_repetitions, key, exercise_ended

        while cap.isOpened():
            # Reading the video frame
            ret, frame = cap.read()
            if not ret:
                break

            # Conversion to grayscale
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False

            # Detection of body landmarks
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Drawing the landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Processing according to selected fiscal year and plan
            if exercicio == 'squat':
                if plano_frontal:
                    frame_bgr = processar_squat_frontal(frame_bgr, results)

                     #Repetitions reset
                    if key == 'r' or key == 'R':
                        repetitions = 0
                        good_repetitions = 0
                        
                          # Resets the repeat count to 0
    
                elif plano_lateral:
                    frame_bgr = processar_squat_lateral(frame_bgr, results)
                    #Repetitions reset
                    if key == 'r' or key == 'R':
                        repetitions = 0
                        good_repetitions = 0 
                        
                          # Resets the repeat count to 0
            elif exercicio == 'deadlift':
                if plano_frontal:
                    frame_bgr = processar_deadlift_frontal(frame_bgr, results)
                     #Repetitions reset
                    if key == 'r' or key == 'R':
                        repetitions = 0
                        good_repetitions = 0
                        
                          # Resets the repeat count to 0
                elif plano_lateral:
                    frame_bgr = processar_deadlift_lateral(frame_bgr, results)
                    # Repetitions reset
                    if key == 'r' or key == 'R':
                        repetitions = 0
                        good_repetitions = 0
                       
                         # Resets the repeat count to 0
                    

            # Display the result
            frame_bgr = exibir_contagem(frame_bgr)  # Add permanent count
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Route for video feed
    @app.route('/video_feed')
    def video_feed():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    

    if __name__ == '__main__':
        app.run(debug=True)

    # release of resources
    cap.release()
    cv2.destroyAllWindows()
