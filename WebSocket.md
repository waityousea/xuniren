## 消息格式

通讯地址：`ws://127.0.0.1:8800/th`

### 发送音频

```
# 发送给api的message数据是str类型
{
    "Topic": "Unreal",
    "Data": {
        "Key": "audio",
        "Value": "C:\samples\sample-1.mp3",
        "Time": 10,
        "Type": "interact"
    }
}
```

| 参数       | 描述             | 类型  | 是否必须 |
| ---------- | ---------------- | ----- | -------- |
| Data.Value | 音频文件绝对路径 | str   | 是       |
| Data.Time  | 音频时长 (秒)    | float | 否       |
| Data.Type  | 发言类型         | str   | 否       |

接口测试场景：Postman

![image-20230420105751124](D:\coding\NeRF\xuniren\ky\img\image-20230420105751124.png)

### 返回视频

```json
# 返回数据的格式，Json
{
    'video': 'data:video/mp4;base64,xxx' 
}
```

| 参数  | 描述                                                       | 类型 | 是否必须 |
| ----- | ---------------------------------------------------------- | ---- | -------- |
| video | base64编码的视频流，前段接收时需要采用base64对视频进行解码 | str  | 是       |
|       |                                                            |      |          |

