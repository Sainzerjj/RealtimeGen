import os
from PIL import Image
import base64
import io
import time
from fastapi.responses import StreamingResponse
import os
import shutil
import json
from item import Item_ori, Item_edit
# 图片转base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

# base64转图片
def convert_base64_to_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return image

def get_pred_ls(output: list[tuple | None]): 
    denoised_prediction = []
    x_t = []
    # 遍历对应文件夹下所有图片得到denoised prediction base64 编码
    for file in output[-1][0]:
        denoised_prediction.append(convert_image_to_base64(file['name']))
    # 遍历对应文件夹下所有图片得到x_t base64 编码
    for file in output[-1][1]:
        x_t.append(convert_image_to_base64(file['name']))
    return {"denoised_prediction": denoised_prediction, "x_t": x_t}

def read_file(hashcode: str):
    denoised_prediction = []
    x_t = []
    image_ls = ["denosed_prediction", "x_t"]
    target_dir = os.path.join("/data/AI_paintings/imgs/ori_imgs", hashcode)
    if not os.path.exists(target_dir):
        target_dir = os.path.join("/data/AI_paintings/imgs/edit_imgs", hashcode)
    for index in range(2):
        next_dir = os.path.join(target_dir, image_ls[index])
        for root, dirs, files in os.walk(next_dir):
            for file in files: 
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    if root.endswith("denosed_prediction"):
                        denoised_prediction.append(convert_image_to_base64(os.path.join(root, file)))
                    elif root.endswith("x_t"):
                        x_t.append(convert_image_to_base64(os.path.join(root, file)))
    return {"denoised_prediction": denoised_prediction, "x_t": x_t}

def save_file(output: list[tuple | None], item: Item_ori | Item_edit):
    # image_files = [f for f in os.listdir(output) 
    #                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # 遍历对应文件夹下所有图片转存到另一个位置, 并删除原文件夹
    image_ls = ["denosed_prediction", "x_t"]
    hashcode = item.hashcode
    for index in range(2):
        if isinstance(item, Item_ori):
            target_dir = os.path.join("/data/AI_paintings/imgs/ori_imgs", hashcode, image_ls[index])
        else:
            target_dir = os.path.join("/data/AI_paintings/imgs/edit_imgs", hashcode, image_ls[index])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if index == 1:
            data = {}
            # 遍历类的属性
            for field in item.__annotations__:
                data[field] = getattr(item, field)
            # 将Python数据转换为JSON字符串
            json_data = json.dumps(data, indent=4)  
            # 将JSON字符串写入文件
            json_path = os.path.join(target_dir, "data.json")
            with open(json_path, "w") as json_file:
                json_file.write(json_data)
        for i, file in enumerate(output[-1][index]):
            image_path = file['name']
            # source_folder = os.path.dirname(image_path)
            # print(source_folder)
            target_path = os.path.join(target_dir, str(i) + ".png")
            shutil.copy(image_path, target_path)
            shutil.rmtree(source_folder)


def save_file_rm(output: list[tuple | None], item: Item_ori | Item_edit):
    # image_files = [f for f in os.listdir(output) 
    #                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # 遍历对应文件夹下所有图片转存到另一个位置, 并删除原文件夹
    if output:
        image_ls = ["denosed_prediction", "x_t"]
        hashcode = item.hashcode
        for index in range(2):
            if isinstance(item, Item_ori):
                target_dir = os.path.join("/data/AI_paintings/imgs/ori_imgs", hashcode, image_ls[index])
            else:
                target_dir = os.path.join("/data/AI_paintings/imgs/edit_imgs", hashcode, image_ls[index])
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            if index == 1:
                data = {}
                # 遍历类的属性
                for field in item.__annotations__:
                    data[field] = getattr(item, field)
                # 将Python数据转换为JSON字符串
                json_data = json.dumps(data, indent=4)  
                # 将JSON字符串写入文件
                json_path = os.path.join(target_dir, "data.json")
                with open(json_path, "w") as json_file:
                    json_file.write(json_data)
            for i, file in enumerate(output[-1][index]):
                image_path = file['name']
                source_folder = os.path.dirname(image_path)
                # print(source_folder)
                target_path = os.path.join(target_dir, str(i) + ".png")
                shutil.copy(image_path, target_path)
                shutil.rmtree(source_folder)
