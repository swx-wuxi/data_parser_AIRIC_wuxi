##############

# *******************创新中心数据采集的.json文件可视化 ***********************
# 测试路径： /home/pc/data_parser_train/output/task_0724_1/20250724152934/scenario_1.json
#  四个场景的总时长（秒）：54 117 109 103

##############

import json
import numpy as np
import matplotlib.pyplot as plt

json_file = "/home/pc/data_parser_train/output/task_0724_1/20250724152934/scenario_1.json"

with open(json_file, "r") as f:
    data = json.load(f)

# scenario_1 两个场景
# 取第二个场景进行测试、图形可视化
for i in range (1,2):
    root = data[i] 
    vm = root.get("vector_map") 

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
    if not np.allclose(ego_past_xy, 0):  
        ax.plot(ego_past_xy[:, 0], ego_past_xy[:, 1], color="red", linewidth=2, label="ego past")


    # ====== 3. 叠加 ego_agent_future ======
    ego_f = np.asarray(root["ego_agent_future"], dtype=float)  # (64,21,7)
    ego_future = ego_f[0]  # 取 batch 0 -> (21,7)法
    ego_future_xy = ego_future[:, :2]

    if not np.allclose(ego_future_xy, 0):  
        ax.plot(ego_future_xy[:, 0], ego_future_xy[:, 1], color="green", linewidth=2, label="ego future")
    
    # # ========4 叠加临车 neighbor_agent_past /future   （问题严重，是数据采集的问题还是解析的问题，不知道）
    # nbr_p = np.asarray(root["neighbor_agents_past"], dtype=float)
    # neighbor_idx = 0    # 第几辆邻车（第 1 辆用 0）
    # nbr_xy = nbr_p[neighbor_idx, :, 0:2] 
    # if not np.allclose(nbr_xy, 0):  
    #     ax.plot(nbr_xy[:, 0], nbr_xy[:, 1], color="pink", linewidth=2, label="neighbor past")
    
    # #  ************** ps 临车未来的轨迹是一个定点，有问题************
    # nbr_f = np.asarray(root["neighbor_agents_future"], dtype=float)
    # nbr_fxy = nbr_f[neighbor_idx, :, 0:2] 
    # print(nbr_fxy[:, 0], nbr_fxy[:, 1])
    # if not np.allclose(nbr_fxy, 0):  
    #     ax.plot(nbr_fxy[:, 0], nbr_fxy[:, 1], color="green", linewidth=2, label="neighbor future")

    #===== 5 画出cross_walks =======
    crosswalks = np.array(vm["map_crosswalks"], dtype=float)  
    for crosswalk in crosswalks:                          
        cros = np.asarray(crosswalk, dtype=float)
        xy = cros[:, :2]
        if np.allclose(xy, 0):                       # 跳过全零
            continue
        ax.plot(xy[:, 0], xy[:, 1], "k-", linewidth=1 ,color='purple') 

    #===== 6 画出route_lanes =======
    routes = np.array(vm["route_lanes"], dtype=float)  
    for route in routes:                          
        route = np.asarray(route, dtype=float)
        xy = route[:, :2]
        if np.allclose(xy, 0):                       # 跳过全零
            continue
        ax.plot(xy[:, 0], xy[:, 1], "k-", linewidth=1 ,color='yellow')  
# ========Final: 画图==========
ax.set_aspect("equal")
ax.set_title("map+ego")
ax.legend()
plt.show()


