# server.py
from flask import Flask, request, jsonify
from flask_sockets import Sockets
import base64
import time
import json
import gevent
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from tools import audio_pre_process, video_pre_process, generate_video,audio_process
import os
import re
import numpy as np

import shutil
import asyncio
import edge_tts
app = Flask(__name__)
sockets = Sockets(app)
video_list = []


async def main(voicename: str, text: str, OUTPUT_FILE):
    communicate = edge_tts.Communicate(text, voicename)

    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass                


def send_information(path, ws):

        print('传输信息开始！')
        #path = video_list[0]
        ''''''
        with open(path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()

        data = {
                'video': 'data:video/mp4;base64,%s' % video_data,
                }
        json_data = json.dumps(data)

        ws.send(json_data)



def txt_to_audio(text_):
    audio_list = []
    audio_path = 'data/audio/aud_0.wav'
    voicename = "zh-CN-YunxiaNeural"
    # 让我们一起学习。必应由 AI 提供支持，因此可能出现意外和错误。请确保核对事实，并 共享反馈以便我们可以学习和改进!
    text = text_
    asyncio.get_event_loop().run_until_complete(main(voicename,text,audio_path))
    audio_process(audio_path)
    
@sockets.route('/dighuman')
def echo_socket(ws):
    # 获取WebSocket对象
    #ws = request.environ.get('wsgi.websocket')
    # 如果没有获取到，返回错误信息
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    # 否则，循环接收和发送消息
    else:
        print('建立连接！')
        while True:
            message = ws.receive()           
            
            if len(message)==0:

                return '输入信息为空'
            else:                                
                txt_to_audio(message)                       
                audio_path = 'data/audio/aud_0.wav'
                audio_path_eo = 'data/audio/aud_0_eo.npy'
                video_path = 'data/video/results/ngp_0.mp4'
                output_path = 'data/video/results/output_0.mp4'
                generate_video(audio_path, audio_path_eo, video_path, output_path)
                video_list.append(output_path)
                send_information(output_path, ws)
                

               

if __name__ == '__main__':

    audio_pre_process()
    video_pre_process()
    
    server = pywsgi.WSGIServer(('127.0.0.1', 8800), app, handler_class=WebSocketHandler)
    server.serve_forever()
    
    