##############

# *******************创新中心数据采集的.json文件可视化 ***********************
# 测试路径： /home/pc/data_parser_train/output/task_0724_1/20250724152934/scenario_1.json
#  四个场景对应的时间（秒）：54 117 109 103

##############

import json
import numpy as np
import matplotlib.pyplot as plt

json_file = "/home/pc/data_parser_train/output/task_0724_1/20250724152934/scenario_1.json"

with open(json_file, "r") as f:
    data = json.load(f)

# A scenario_0
for i in range (0,2):
    root = data[i] 
    vm = root.get("vector_map") or root.get("vector_Map") or root

    #===== 1 画最基本的map_lanes =======
    lanes = np.array(vm["map_lanes"], dtype=float)  # 可能是 (40,50,7) 或 (64,40,50,7)
    if lanes.ndim == 4 and lanes.shape[-1] == 7:     # 有 batch 的情况
        lanes = lanes[0]                              # 取第 0 个 batch，-> (40,50,7)

    fig, ax = plt.subplots(figsize=(8, 8))
    for lane in lanes:                                # lane: (50,7) 或（异常）(N,7)
        lane = np.asarray(lane, dtype=float)
        if lane.ndim != 2 or lane.shape[1] < 2:      # 防御：不是 (N,7) 就跳过
            continue
        xy = lane[:, :2]
        if np.allclose(xy, 0):                       # 跳过全零车道
            continue
        ax.plot(xy[:, 0], xy[:, 1], "k-", linewidth=1 ,color='blue')    

    # ====== 2. 叠加 ego_agent_past ======
    ego_p = np.asarray(root["ego_agent_past"], dtype=float)  # (64,21,7)
    ego_past = ego_p[0]  # 取 batch 0 -> (21,7)
    ego_past_xy = ego_past[:, :2]
    print(ego_past_xy)
    if not np.allclose(ego_past_xy, 0):  
        ax.plot(ego_past_xy[:, 0], ego_past_xy[:, 1], color="red", linewidth=2, label="ego past")


    # ====== 3. 叠加 ego_agent_future ======
    ego_f = np.asarray(root["ego_agent_future"], dtype=float)  # (64,21,7)
    ego_future = ego_f[0]  # 取 batch 0 -> (21,7)法
    ego_future_xy = ego_future[:, :2]

    if not np.allclose(ego_future_xy, 0):  
        ax.plot(ego_future_xy[:, 0], ego_future_xy[:, 1], color="green", linewidth=2, label="ego future")
# 111

# ========Final: 画图==========
ax.set_aspect("equal")
ax.set_title("map_lanes (x,y) + ego trajectory")
ax.legend()
plt.show()


