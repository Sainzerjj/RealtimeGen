import asyncio
import websockets
import requests
import os
import json
from back_utils import *
# WebSocket连接URL
websocket_url = "ws://localhost:8001/edit_generate"

# 获取进度元素
progress_element = None

# WebSocket消息处理函数
# async def websocket_handler():
#     async with websockets.connect(websocket_url) as websocket:
#         while True:
#             pred_data = await websocket.recv()  # 接收WebSocket消息
#             # 更新页面上的进度
#             print(f"Prediction: {pred_data}%")
    # try:
    #     async with websockets.connect(websocket_url) as websocket:
    #         while True:
    #             pred_data = await websocket.recv()  # 接收WebSocket消息
    #             # 更新页面上的进度
    #             print(f"Prediction: {pred_data}%")
    # except websockets.exceptions.ConnectionClosedOK:
    #     print("WebSocket connection closed")

# 发起 POST 请求
async def send_post_request():
    async with websockets.connect(websocket_url, max_size = 30*800*800) as websocket:
        
        # 插入的图片(base64格式)
        # "editted_image":convert_image_to_base64("/data/AI_paintings/imgs/ori_imgs/2023081601/x_t/0.png"),
        # 插入上述图片的时间节点，要小于对应文生图的步数
        # "insert_time":3
        editted_image = convert_image_to_base64("/home/amax/Documents/zsz/AI-painting/imgs/ori_imgs/string/x_t/0.png")
        # Item_edit
        data = '{"__type__":"Item_edit","hashcode":"20230826","text":"a white cat","samples":1,"steps":10,"scale":7.5,"seed":1024,"width":768,"height":768,"scheduler_dd":"DDIM","editted_image":"","insert_time":2}'
        data_dict = json.loads(data)
        data_dict["editted_image"] = editted_image
        new_data = json.dumps(data_dict)
        # response = requests.post("http://localhost:8002/ori_generate", json=data, headers=headers)
        # if response.status_code == 200:
        #     data = response.json()
        #     print(data)
        # else:
        #     print("Error:", response.status_code)
        try:
            await websocket.send(new_data)
            while True:
                # 持续收到生成的图片地址
                pred = await websocket.recv()
                # pred = json.loads(pred)
                print(pred) # http得到地址 
                # 如果用户选择了停止，就发送interrupt信号。此处使用其他停止信号。
                # if any(['stop' in tmp for tmp in os.listdir()]):
                #     await websocket.send("interrupt")
                # else: # 否则继续
                #     await websocket.send("get")
                if isinstance(pred, str) and pred == "finishes": # 生成完成
                    break
        except websockets.exceptions.ConnectionClosedOK:
            print("WebSocket connection closed")
        finally:
            # 关闭WebSocket连接
            
            await websocket.close()

if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.get_event_loop()

    # 创建任务列表
    tasks = []

    # 创建WebSocket任务
    # tasks.append(asyncio.ensure_future(websocket_handler()))

    # 创建发送POST请求的任务
    # tasks.append(asyncio.ensure_future(loop.run_in_executor(None, send_post_request)))

    # 
    # 执行任务
    loop.run_until_complete(send_post_request())