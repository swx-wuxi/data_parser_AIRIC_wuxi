import os
import json
import numpy as np

# ==== 你需要自己修改这几个变量 ====
output_base_dir = "/home/WorkSpace/output"  # 保存文件的根目录
task_folder = "task_0724_2"                      # 对应的任务文件夹
timestamp = "20250724153945"              # 保存时用的时间戳
scene_idx = 0                             # 要查看的场景索引
# ====================================

# 组合文件路径
file_path = os.path.join(
    output_base_dir,
    task_folder,
    timestamp,
    f"scenario_{scene_idx}.json"
)

def get_shape(value):
    """尝试获取对象的维度信息"""
    try:
        arr = np.array(value)
        return arr.shape
    except Exception:
        return None

def describe_field(name, value):
    """打印字段信息 + 维度"""
    if isinstance(value, list):
        print(f"  - {name}: list (长度={len(value)})")
        if len(value) > 0:
            shape = get_shape(value)
            if shape:
                print(f"    📐 维度: {shape}")
            print(f"    示例元素类型: {type(value[0]).__name__}")
    elif isinstance(value, dict):
        print(f"  - {name}: dict (key数={len(value)})")
        keys = list(value.keys())
        print(f"    示例keys: {keys[:5]}")
    else:
        print(f"  - {name}: {type(value).__name__} = {value}")

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"❌ 找不到文件: {file_path}")
else:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        print(f"📌 JSON 是一个列表，共 {len(data)} 个元素")
        for idx, element in enumerate(data):
            print(f"\n=== 元素 {idx} ===")
            if isinstance(element, dict):
                for field_name, field_value in element.items():
                    describe_field(field_name, field_value)
            else:
                print(f"  元素类型: {type(element).__name__}")
