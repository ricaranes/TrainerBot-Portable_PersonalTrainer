from flask import Flask, render_template, Response
import os
os.environ['PATH'] += ':/usr/local/bin'
import cv2
import mediapipe as mp
import time
import subprocess
import pytesseract
mp_pose = mp.solutions.pose

def analisar_frame(frame):
    # Converter a imagem para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Realizar a detecção de texto na imagem usando OCR (Optical Character Recognition)
    text = pytesseract.image_to_string(gray)

    # Verificar se a mensagem de erro está presente no texto
    if "Error:" in text:
        return False  # Repetição com erros
    else:
        return True  # Repetição bem executada
    
    

def text_to_speech(text):
    # Verificação do tempo para o feedback de áudio
    current_time = time.time()
    if current_time - text_to_speech.last_feedback_time >= text_to_speech.feedback_interval:
        text_to_speech.last_feedback_time = current_time
        subprocess.call(["say", text])


def exibir_contagem(frame_bgr):
    import __main__ as main  # Importe o módulo principal (__main__)

    # Acesse a variável global repetitions do módulo principal
    repetitions = main.repetitions

    # Resto do código da função...

    cv2.putText(frame_bgr, f"Repetitions: {repetitions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return frame_bgr