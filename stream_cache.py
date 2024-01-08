"""
Create by fay.xszyou on 20230706.
缓冲器
"""
from io import BytesIO
import threading
import functools

def synchronized(func):
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    with self.lock:
      return func(self, *args, **kwargs)
  return wrapper

class StreamCache:
    def __init__(self, maxbytes):
        self.lock = threading.Lock()
        self.list = []
        self.writeSeek = 0
        self.readSeek = 0
        self.maxbytes = maxbytes
        self.idle = 0
        
    @synchronized
    def write(self, bs):
        # print("写:{},{}".format(len(bs),bs), end=' ')
        if self.idle >= self.maxbytes:
            print("缓存区不够用")
        if self.writeSeek + len(bs) <= self.maxbytes:
            self.list[self.writeSeek : self.writeSeek + len(bs)]  = bs
            self.writeSeek = self.writeSeek + len(bs)
        else:
            self.list[self.writeSeek : self.maxbytes] = bs[0:self.maxbytes - self.writeSeek]
            self.list[0 : len(bs) - (self.maxbytes - self.writeSeek)] = bs[self.maxbytes - self.writeSeek:]
            self.writeSeek = len(bs) - (self.maxbytes - self.writeSeek)
        self.idle += len(bs)

        if self.writeSeek >= self.maxbytes:
            self.writeSeek = 0

    
    @synchronized
    def read(self, length, exception_on_overflow = False):
        if self.idle < length:
            return None
        # print("读:{}".format(length), end=' ')
        if self.readSeek + length <= self.maxbytes:
            bs = self.list[self.readSeek : self.readSeek + length]
            self.readSeek = self.readSeek + length
        else:
            bs = self.list[self.readSeek : self.maxbytes]
            bs[self.maxbytes - self.readSeek : length] = self.list[0:length - (self.maxbytes - self.readSeek)]
            self.readSeek =  length - (self.maxbytes - self.readSeek)

        self.idle -= length
        if self.readSeek >= self.maxbytes:
           self.readSeek = 0
        return bs

    @synchronized
    def clear(self):
        self.list = []
        self.writeSeek = 0
        self.readSeek = 0
        self.idle = 0

if __name__ == '__main__':
    streamCache = StreamCache(50)
    streamCache.write([[[95,181 , 93],[ 91, 177 , 89],[ 86, 172 , 84],[ 86 ,172 , 84],[ 85, 171  ,83], [ 84 ,170,  82],[ 86, 172 , 84],[ 91, 177  ,89],[ 93, 179  ,91]]])
    streamCache.write(b'\x03\x04\x00')
    print(streamCache.read(1))
    print(streamCache.read(3))
    streamCache.write(b'\x05\x06')
    print(streamCache.read(2))
    print(streamCache.read(2))
    print(streamCache.read(3))