from flask import Flask, render_template, Response
import os
os.environ['PATH'] += ':/usr/local/bin'
import cv2
import mediapipe as mp
import time
import subprocess
import pytesseract

from functions import text_to_speech, exibir_contagem, analisar_frame
repetitions = 0
good_repetitions = 0
contagem_ativa = False
frame_sem_erros = True

# MediaPipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def processar_squat_frontal(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros

    # Verifica se há resultados disponíveis
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Lógica para o plano frontal
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
    
        # Verificação de erro: joelhos para dentro
        left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
        left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
        right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x

        if left_knee_x < left_ankle_x or right_knee_x > right_ankle_x:
            cv2.putText(frame_bgr, "Error: Knees in", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame_sem_erros = False

        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Verificação de erro: joelhos acima dos quadris
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
        left_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        right_knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y

        if left_knee_y >= left_hip or right_knee_y >= right_hip:
            cv2.putText(frame_bgr, "Error: Knees above hips ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame_sem_erros = False

        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Contagem de repetições    
        if left_knee_y < left_hip or right_knee_y < right_hip:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Salvar o frame em um arquivo
                cv2.imwrite("frame.jpg", frame_bgr)

                # Pausa de 3 segundos após cada repetição
                time.sleep(0.1)

                # Realizar a análise do frame gravado
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Verificar se a repetição gravada foi bem feita
                if repeticao_bem_feita:
                    # Incrementar a contagem de repetições bem executadas
                    good_repetitions += 1
                    # Imprimir feedback positivo
                    print("Repetição bem executada!")
                else:
                    # Imprimir feedback negativo
                    print("Repetição com erros!")

        else:
            contagem_ativa = False

    return frame_bgr


# Função para processar o plano frontal do exercício "deadlift"
def processar_deadlift_frontal(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros
    # Check if there are results available
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Logic for the front view
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
       
         # Shoulder lift
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        mouth_right = results.pose_landmarks.landmark[9]
        mouth_left = results.pose_landmarks.landmark[10]

        shoulder_to_mouth_distance = abs(shoulder_right.y - mouth_right.y)  # Vertical distance between right shoulder and right mouth
        shoulder_to_mouth_distance += abs(shoulder_left.y - mouth_left.y)  # Add vertical distance between left shoulder and left mouth
        shoulder_to_mouth_distance /= 2  # Calculate the average of the distances

        if shoulder_to_mouth_distance < 0.08:  # Reference value for shoulder elevation detection
            cv2.putText(frame_bgr, "Error: High Shoulders", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Knees in
        left_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x
        right_ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
        left_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x
        right_knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x
        
        threshold = 0.02

        if left_knee_x < left_ankle_x - threshold or right_knee_x > right_ankle_x + threshold:
            cv2.putText(frame_bgr, "Error: Knees in", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # contagem de repetições
        # Obtém as coordenadas das landmarks dos pulsos e joelhos
        right_wrist_y = results.pose_landmarks.landmark[16].y  # Pulso direito
        left_wrist_y = results.pose_landmarks.landmark[15].y  # Pulso esquerdo
        right_knee_y = results.pose_landmarks.landmark[26].y  # Joelho direito
        left_knee_y = results.pose_landmarks.landmark[25].y  # Joelho esquerdo

        # Verifica se as landmarks dos pulsos estão iguais ou abaixo dos joelhos
        if right_wrist_y <= right_knee_y or left_wrist_y <= left_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Salvar o frame em um arquivo
                cv2.imwrite("frame.jpg", frame_bgr)

                # Pausa de 0.1 segundos após cada repetição
                time.sleep(0.1)

                # Realizar a análise do frame gravado
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Verificar se a repetição gravada foi bem feita
                if repeticao_bem_feita:
                    # Incrementar a contagem de repetições bem executadas
                    good_repetitions += 1
                    # Imprimir feedback positivo
                    print("Repetição bem executada!")
                else:
                    # Imprimir feedback negativo
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
            
    return frame_bgr



# Função para processar o plano lateral do exercício "squat"
def processar_squat_lateral(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros
    # Verifica se há resultados disponíveis
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Lógica para o plano lateral
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
        # Lógica de detecção de erros para o plano lateral
        left_knee = results.pose_landmarks.landmark[25].x
        right_knee = results.pose_landmarks.landmark[26].x
        left_ankle = results.pose_landmarks.landmark[31].x
        right_ankle = results.pose_landmarks.landmark[32].x
        left_shoulder = results.pose_landmarks.landmark[11].x
        right_shoulder = results.pose_landmarks.landmark[12].x

        # Verificação de erro: joelhos a passarem a ponta do pé
        if left_knee < left_ankle or right_knee < right_ankle:
            cv2.putText(frame_bgr, "Error: knees too far forward", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Verificação de erro: ombros passando dos joelhos
        if left_shoulder < left_knee or right_shoulder < right_knee:
            cv2.putText(frame_bgr, "Error: Trunk leaning too far forward", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # contagem de repetições
        left_hip_y = results.pose_landmarks.landmark[23].y  # Número da landmark correspondente à anca esquerda
        right_hip_y = results.pose_landmarks.landmark[24].y  # Número da landmark correspondente à anca direita
        left_knee_y = results.pose_landmarks.landmark[25].y  # Número da landmark correspondente ao joelho esquerdo
        right_knee_y = results.pose_landmarks.landmark[26].y  # Número da landmark correspondente ao joelho direito

        
        if left_knee_y >= left_hip_y or right_knee_y >= right_hip_y:
            cv2.putText(frame_bgr, "Error: Knees above hips ", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            frame_sem_erros = False

        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            

        # Verificar se a anca está abaixo do joelho
        if left_hip_y < left_knee_y or right_hip_y < right_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Salvar o frame em um arquivo
                cv2.imwrite("frame.jpg", frame_bgr)

                # Pausa de 0.1 segundos após cada repetição
                time.sleep(0.1)

                # Realizar a análise do frame gravado
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Verificar se a repetição gravada foi bem feita
                if repeticao_bem_feita:
                    # Incrementar a contagem de repetições bem executadas
                    good_repetitions += 1
                    # Imprimir feedback positivo
                    print("Repetição bem executada!")
                else:
                    # Imprimir feedback negativo
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
        
    return frame_bgr



def processar_deadlift_lateral(frame_bgr, results):
    global repetitions, good_repetitions, contagem_ativa, frame_sem_erros
    # Verifica se há resultados disponíveis
    if results is None or not results.pose_landmarks:
        return frame_bgr

    # Lógica para o plano lateral
    if results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility > 0.5:
        # Costas arqueadas
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        mouth_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]
        mouth_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        shoulder_to_mouth_distance = abs(shoulder_right.y - mouth_right.y)  # Distância vertical entre ombro direito e boca direita
        shoulder_to_mouth_distance += abs(shoulder_left.y - mouth_left.y)  # Adiciona distância vertical entre ombro esquerdo e boca esquerda
        shoulder_to_mouth_distance /= 2  # Calcula a média das distâncias

        if shoulder_to_mouth_distance < 0.08 and nose.y > shoulder_right.y and nose.y > shoulder_left.y:
            cv2.putText(frame_bgr, "Error: Costas arqueadas", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Joelhos adiantados  

        left_knee = results.pose_landmarks.landmark[25].x
        right_knee = results.pose_landmarks.landmark[26].x
        left_ankle = results.pose_landmarks.landmark[31].x
        right_ankle = results.pose_landmarks.landmark[32].x

        knee_threshold = 0.01  # Limite de diferença entre os joelhos e a ponta do pé

        # Verificação de erro: joelhos a passarem a ponta do pé
        if left_knee < left_ankle - knee_threshold or right_knee < right_ankle - knee_threshold:
            cv2.putText(frame_bgr, "Error: knees too far forward", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Excessive trunk forward lean
        shoulder_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        knee_right = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        knee_left = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]

        shoulder_knee_threshold = 0.06  # Horizontal difference threshold between shoulders and knees

        if abs(shoulder_right.x - knee_right.x) > shoulder_knee_threshold or abs(shoulder_left.x - knee_left.x) > shoulder_knee_threshold:
            cv2.putText(frame_bgr, "Warning: Straighten your trunk", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_bgr, "Great Job! Nothing to add", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # contagem de repetições
        # Obtém as coordenadas das landmarks dos pulsos e joelhos
        right_wrist_y = results.pose_landmarks.landmark[16].y  # Pulso direito
        left_wrist_y = results.pose_landmarks.landmark[15].y  # Pulso esquerdo
        right_knee_y = results.pose_landmarks.landmark[26].y  # Joelho direito
        left_knee_y = results.pose_landmarks.landmark[25].y  # Joelho esquerdo

        # Verifica se as landmarks dos pulsos estão iguais ou abaixo dos joelhos
        if right_wrist_y <= right_knee_y or left_wrist_y <= left_knee_y:
            if not contagem_ativa:
                repetitions += 1
                contagem_ativa = True

                # Salvar o frame em um arquivo
                cv2.imwrite("frame.jpg", frame_bgr)

                # Pausa de 0.1 segundos após cada repetição
                time.sleep(0.1)

                # Realizar a análise do frame gravado
                if os.path.exists("frame.jpg"):
                    frame = cv2.imread("frame.jpg")
                    repeticao_bem_feita = analisar_frame(frame)
                else:
                    repeticao_bem_feita = False

                # Verificar se a repetição gravada foi bem feita
                if repeticao_bem_feita:
                    # Incrementar a contagem de repetições bem executadas
                    good_repetitions += 1
                    # Imprimir feedback positivo
                    print("Repetição bem executada!")
                else:
                    # Imprimir feedback negativo
                    print("Repetição com erros!")

        else:
            contagem_ativa = False
            

    return frame_bgr