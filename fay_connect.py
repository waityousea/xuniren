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
import cv2
import pygame

video_list = []

fay_ws = None

def connet_fay():
    def connect():
        ws_url = "ws://127.0.0.1:10002"
        fay_ws = websocket.WebSocketApp(ws_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close)
        fay_ws.on_open = on_open
        fay_ws.run_forever()
    connect()
if __name__ == '__main__':

    threading.Thread(target=connet_fay, args=[]).start()


    
    

