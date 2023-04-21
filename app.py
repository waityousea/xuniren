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
app = Flask(__name__)
sockets = Sockets(app)
video_list = []





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



@sockets.route('/th')
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
                                
                message=message.replace("\\","/")    
                print('message:', message,type(message))
                message = eval(message)
                aud_dir = message["Data"]["Value"]
                basedir = ""
                for i in aud_dir.split("/"):
                    basedir = os.path.join(basedir,i)
                basedir = basedir.replace(":",":\\")   
                num = 1
                new_path = r'./data/audio/aud_%d.wav'%num  #新路径                
                old_path = basedir                
                shutil.copy(old_path, new_path)            
                audio_path = 'data/audio/aud_%d.wav' % num
                audio_process(audio_path)
                audio_path_eo = 'data/audio/aud_%d_eo.npy' % num
                video_path = 'data/video/results/ngp_%d.mp4' % num
                output_path = 'data/video/results/output_%d.mp4' % num
                generate_video(audio_path, audio_path_eo, video_path, output_path)
                video_list.append(output_path)
                send_information(output_path, ws)
                

               

if __name__ == '__main__':

    audio_pre_process()
    video_pre_process()
    
    server = pywsgi.WSGIServer(('127.0.0.1', 8800), app, handler_class=WebSocketHandler)
    server.serve_forever()
    
    