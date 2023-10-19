from gradio_client import Client
from back_utils import *
from item import Item_ori, Item_edit
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from typing import Union
import time
from fastapi.responses import StreamingResponse
import asyncio
from fastapi import FastAPI, WebSocket, Response, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from websockets.exceptions import WebSocketException
# from websockets.exceptions import ConnectionClosed
# from fastapi import WebSocketDisconnect
import websockets
import random
from fastapi.responses import FileResponse


app = FastAPI()
# 用于连接gradio app服务的api
# https://7d14a26a11770e9d12.gradio.live
client = Client(src="http://0.0.0.0:8000", serialize=False, output_dir="./save_img", )  # connecting to a temporary Gradio share URL
# s = client.view_api(all_endpoints=True)  # 查看api endpoint
# WebSocket连接集合
# websocket_connections = []

# post_request_completed = False

# WebSocket路由
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     global post_request_completed
#     await websocket.accept()
#     websocket_connections.append(websocket)
#     try:
#         while True:
#             # await asyncio.sleep(1)  # 模拟处理进度

#             if post_request_completed:  # 检查POST请求是否完成
#                 break


#             # progress = get_progress()  # 获取进度信息
#             # await websocket.send_text(str(progress))  # 发送进度信息
#     except websockets.exceptions.ConnectionClosedOK:
#         print("WebSocket connection closed")
#     finally:
#         # 关闭WebSocket连接
#         # await websocket.close()
#         websocket_connections.remove(websocket)


# SD正常功能输入条件生成图像
# @app.post("/ori_generate/")
@app.websocket("/ori_generate")
# async def Ori_Gen(item : Item_ori, response: Response):
#     #处理跨域问题
#     response.headers["Access-Control-Allow-Origin"] = "*"
async def Ori_Gen(websocket: WebSocket):
    global post_request_completed

    await websocket.accept()
    # 接受本次生成的json格式字符串
    json_string = await websocket.receive_text()
    # 变成Item_ori
    item = Item_ori.validate(json.loads(json_string))
    print(type(item))
    # item = json.loads(json_string)
    post_request_completed = False
    
    text = item.text
    samples = item.samples
    steps = item.steps
    scale = item.scale
    seed = item.seed
    width = item.width
    height = item.height
    scheduler_dd = item.scheduler_dd
    # 提交参数给SD模型端
    job = client.submit(text, samples, steps, scale, 
                            seed, height, width, scheduler_dd, fn_index=2)  # runs the prediction in a background thread
                            
    # 得到返回的输出
    output = job.outputs()
    # job对象运行需要一定时间（取决于生成步数大小），但程序会继续往下执行，可以等待job对象运行完成，
    # 但系统实时性无法保证，所以这里采用轮询的方式，每隔2秒检查一次job对象是否运行完成，output长度是实时变化的
    # 以生成三步图像为例，job对象运行完成后，output的长度为5，所以这里轮询的条件为len(output) <= steps + 2
    # output保存位置在由上面Client实例化output_dir指定的文件夹下，文件名为job对象的id(默认为uuid)   
    # 写一段监听代码，用于实时监听output是否发生变化，如果发生变化，就将变化的图片返回给前端
    try:
        while True:
            # if 0 < len(output) < steps + 2:
            await asyncio.sleep(1)
            # print(len(output))
            # 得到denoised prediction和x_t的base64编码的字典
            if output:
                pred_ls = get_pred_ls(output)
                # save_file(output, item)
                pred_ls_str = json.dumps(pred_ls)
                # 对应步数图片保存完毕后，返回给前端确认消息，允许前端根据http路径读取图片
                await websocket.send_text(pred_ls_str)
                # await websocket.send_text(f"Output of Step {len(output) - 1} have Saved!")
                await asyncio.sleep(5)
                # update_progress(pred_ls)
                # yield "One post arrived!"
                # yield pred_ls
            if len(output) == steps + 2:
                post_request_completed = True
                print("send finish message")
                await websocket.send_text("finish")
                break
        
    except WebSocketDisconnect:
        print(F"WebSocket of *** Disconnect")
    finally:
        save_file_rm(output, item)
        await websocket.close()
        

    return {"data": "Post Finish!"}
    #yield "finished!"
        # yield pred_ls
    # return {"data": [result async for result in gen_ori(item)]}

# SD附加功能：上传编辑后的图片，并指定对应插入的时间节点
@app.websocket("/edit_generate")
async def Edit_Gen(websocket : WebSocket):
    #处理跨域问题
    # response.headers["Access-Control-Allow-Origin"] = "*"
    # async def gen_edit(item : Item_edit):
    global post_request_completed
    await websocket.accept()
    # 接受本次生成的json格式字符串
    json_string = await websocket.receive_text()
    # 变成Item_ori
    item = Item_edit.validate(json.loads(json_string))
    # post_request_completed = False
    
    text = item.text
    samples = item.samples
    steps = item.steps
    scale = item.scale
    seed = item.seed
    width = item.width
    height = item.height
    scheduler_dd = item.scheduler_dd
    # 编辑后的图像以及插入时间
    editted_image: Union[str, None] = None
    insert_time = item.insert_time
    
    editted_image = convert_base64_to_image(item.editted_image)
    # 提交参数给SD模型端
    job = client.submit(text, samples, steps, scale, 
                            seed, height, width, scheduler_dd, editted_image, insert_time, fn_index=3)  # runs the prediction in a background thread
                            
    # 得到返回的输出
    output = job.outputs()
    # job对象运行需要一定时间（取决于生成步数大小），但程序会继续往下执行，可以等待job对象运行完成，
    # 但系统实时性无法保证，所以这里采用轮询的方式，每隔2秒检查一次job对象是否运行完成，output长度是实时变化的
    # 以生成三步图像为例，job对象运行完成后，output的长度为5，所以这里轮询的条件为len(output) <= steps + 2
    # output保存位置在由上面Client实例化output_dir指定的文件夹下，文件名为job对象的id(默认为uuid)   
    # 写一段监听代码，用于实时监听output是否发生变化，如果发生变化，就将变化的图片返回给前端
    try:
        while True:
            # if 0 < len(output) < steps + 2:
            await asyncio.sleep(1)
            # print(len(output))
            # 得到denoised prediction和x_t的base64编码的字典
            if output:
                pred_ls = get_pred_ls(output)
                # save_file(output, item)
                pred_ls_str = json.dumps(pred_ls)
                # 对应步数图片保存完毕后，返回给前端确认消息，允许前端根据http路径读取图片
                await websocket.send_text(pred_ls_str)
                await asyncio.sleep(5)
                # await websocket.send_text(f"Output of Step {len(output) - 1} have Saved!")
                # yield pred_ls
            if len(output) == steps + 2:
                print("finish")
                post_request_completed = True
                await websocket.send_text("finish")
                break
        
    except WebSocketDisconnect:
        print(F"WebSocket of *** Disconnect")
    finally:
        save_file_rm(output, item)
        await websocket.close()

    return {"data": "Post Finish!"}
   

# SD附加功能：查看历史记录
@app.post("/history/")
async def history(hashcode: str, response: Response):
    #处理跨域问题
    response.headers["Access-Control-Allow-Origin"] = "*"
    # 根据前端提供的hashcode读取文件
    select_dict = read_file(hashcode)
    # select_dict = {"denoised_prediction": denoised_prediction, "x_t": x_t}
    return select_dict


if __name__=='__main__':
    uvicorn.run(app='server:app', host="0.0.0.0", port=8000, reload=True)

