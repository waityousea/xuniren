from  stream_cache import StreamCache

video_stream_cache = None
video_len = 0

def write(frame):
    global video_stream_cache
    video_stream_cache.write(frame)

def start():
    global video_stream_cache
    video_stream_cache = StreamCache(10240)

def read():
    global video_stream_cache
    if video_stream_cache.idle > 0:
        video_imgs = video_stream_cache.read(1)
        if video_imgs:
            return video_imgs
    else:
        return []
        
                # for img in video_imgs:
                #     _, buffer = cv2.imencode('.jpg', img)
                #     encoded_img = base64.b64encode(buffer).decode('utf-8')
                    # ws_server.new_instance().add_cmd({"reply": {"img": encoded_img}})

def get_idle():
    global video_stream_cache
    return  video_stream_cache.idle

def set_video_len(n):
    global video_len
    video_len = n

def get_video_len():
    global video_len
    return  video_len