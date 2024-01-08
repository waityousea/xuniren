"""
郭泽斌于2023.04.29参照app.py创建，用于连接github开源项目 Fay 数字人
"""

import base64
import time
import json
import gevent
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from tools import audio_pre_process, video_pre_process, generate_video, audio_process
import os
import re
import numpy as np
import threading
import websocket
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
import cv2
import pygame
import hashlib
import video_stream

video_list = []
audio_paths = []
fay_ws = None
video_cache = {}

#增加MD5音频标记，避免重复生成视频
def hash_file_md5(filepath):
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(65536)  # Read in 64kb chunks
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

def connet_fay():
    global video_list
    global video_cache
    global fay_ws
    global audio_paths

    def on_message(ws, message):
        if "audio" in message:
            message_dict = json.loads(message)
            aud_dir = message_dict["Data"]["Value"]
            aud_dir = aud_dir.replace("\\", "/")
            print('message:', aud_dir, type(aud_dir))
            basedir = ""
            for i in aud_dir.split("/"):
                basedir = os.path.join(basedir,i)
            basedir = basedir.replace(":",":\\")   
            num = time.time()
            new_path = r'./data/audio/aud_%d.wav'%num  #新路径                
            old_path = basedir                
 
            convert_mp3_to_wav(old_path, new_path) 
            audio_hash = hash_file_md5(new_path)
            audio_paths.append(new_path)
            # if audio_hash in video_cache:
            #     video_list.append({"video": video_cache[audio_hash], "audio": new_path})
            #     ret, frame = cap.read()

            #     print("视频已存在，直接播放。")
            # else:
            audio_path = 'data/audio/aud_%d.wav' % num
            audio_process(audio_path)
            audio_path_eo = 'data/audio/aud_%d_eo.npy' % num
            video_path = 'data/video/results/ngp_%d.mp4' % num
            output_path = 'data/video/results/output_%d.mp4' % num
            generate_video(audio_path, audio_path_eo, video_path, output_path)
            video_list.append({"video" : output_path, "audio" : new_path})
            video_cache[audio_hash] = output_path

    def on_error(ws, error):
        print(f"Fay Error: {error}")
        reconnect()

    def on_close(ws):
        print("Fay Connection closed")
        reconnect()

    def on_open(ws):
        print("Fay Connection opened")

    def connect():
        global fay_ws
        ws_url = "ws://127.0.0.1:10002"
        fay_ws = websocket.WebSocketApp(ws_url,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close)
        fay_ws.on_open = on_open
        fay_ws.run_forever()

    def reconnect():
        global fay_ws
        fay_ws = None
        time.sleep(5)  # 等待一段时间后重连
        connect()

    connect()

def convert_mp3_to_wav(input_file, output_file):
    audio = AudioSegment.from_mp3(input_file)
    audio.export(output_file, format='wav')


def play_video():
    global video_list
    global audio_paths
    audio_path = None
    frame = None
    _, frame = cv2.VideoCapture("data/pretrained/train.mp4").read()
    while True:
        if video_stream.get_idle() > int(video_stream.get_video_len() / 3):
            if len(audio_paths)>0:
                audio_path = audio_paths.pop(0)
                print(audio_path)
                threading.Thread(target=play_audio, args=[audio_path]).start()  # play audio
            i = video_stream.get_video_len()
            video_stream.set_video_len(0)
            #循环播放视频帧
            while True:
                imgs = video_stream.read()
                if len(imgs) > 0:
                    frame = imgs[0]
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Fay-2d', frame)
                    # 等待 38 毫秒
                    cv2.waitKey(38)
                    i = i - 1
                elif i == 0:
                    break
        else:
            cv2.imshow('Fay-2d', frame)
            # 等待 38 毫秒
            cv2.waitKey(38)

def play_audio(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
if __name__ == '__main__':

    audio_pre_process()
    video_pre_process()
    video_stream.start()
    threading.Thread(target=connet_fay, args=[]).start()
    play_video()


    
    

