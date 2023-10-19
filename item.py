from pydantic import BaseModel
from typing import Union
# 请求体格式
# SD普通生成的请求体
class Item_ori(BaseModel):
    # 前端提供和保存的指定参数，用于确定保存和读取图片位置
    hashcode:  Union[str, None] = None
    # 前端输入的模型参数
    text: Union[str, None] = None
    samples: int = 1
    steps: int = 25
    scale: float = 7.5
    seed: int = 1024
    width: int = 768
    height: int = 768
    scheduler_dd: str = "DDIM"

# 基于中间编辑的SD生成的请求体
class Item_edit(BaseModel):
    # 前端提供和保存的指定参数，用于确定保存和读取图片位置
    hashcode:  Union[str, None] = None
    # 前端输入的模型参数
    text: Union[str, None] = None
    samples: int = 1
    steps: int = 25
    scale: float = 7.5
    seed: int = 1024
    width: int = 768
    height: int = 768
    scheduler_dd: str = "DDIM"
    editted_image: Union[str, None] = None
    insert_time: int = 10