import json
import numpy as np
import matplotlib.pyplot as plt

json_file = "/home/pc/data_parser_train/output/task_0724_1/20250724152934/scenario_1.json"

with open(json_file, "r") as f:
    data = json.load(f)

# 统一成场景列表
scenes = data if isinstance(data, list) else [data]

def take_batch(arr):
    a = np.asarray(arr, dtype=float)
    # (64, T, D) -> 取 batch 0；否则原样返回
    return a[0] if (a.ndim >= 2 and a.shape[0] == 64) else a

def all_zero(xy):
    return np.allclose(xy, 0)

fig, ax = plt.subplots(figsize=(8, 8))

added_lane_label   = False
added_past_label   = False
added_future_label = False

for i, scene in enumerate(scenes):
    vm = scene.get("vector_map") or scene.get("vector_Map") or scene

    # ===== 画 map_lanes（蓝色，跳全零）=====
    lanes = np.array(vm["map_lanes"], dtype=float)  # (40,50,7) 或 (64,40,50,7)
    if lanes.ndim == 4 and lanes.shape[-1] == 7:
        lanes = lanes[0]  # 取 batch 0 -> (40,50,7)

    for lane in lanes:  # lane: (50,7)
        lane = np.asarray(lane, dtype=float)
        if lane.ndim != 2 or lane.shape[1] < 2:
            continue
        xy = lane[:, :2]
        if all_zero(xy):
            continue
        ax.plot(
            xy[:, 0], xy[:, 1],
            color="blue", linewidth=1,
            label=None if added_lane_label else f"map_lanes (scene {i})"
        )
        added_lane_label = True

    # ===== 叠加 ego_agent_past（红）=====
    if "ego_agent_past" in scene:
        ego_p = take_batch(scene["ego_agent_past"])  # (21,7) 或 (T,7)
        if ego_p.ndim == 2 and ego_p.shape[1] >= 2:
            ego_p_xy = ego_p[:, :2]
            if not all_zero(ego_p_xy):
                ax.plot(
                    ego_p_xy[:, 0], ego_p_xy[:, 1],
                    color="red", linewidth=2,
                    label=None if added_past_label else "ego past (all scenes)"
                )
                added_past_label = True

    # ===== 叠加 ego_agent_future（绿）=====
    if "ego_agent_future" in scene:
        ego_f = take_batch(scene["ego_agent_future"])
        if ego_f.ndim == 2 and ego_f.shape[1] >= 2:
            ego_f_xy = ego_f[:, :2]
            if not all_zero(ego_f_xy):
                ax.plot(
                    ego_f_xy[:, 0], ego_f_xy[:, 1],
                    color="green", linewidth=2,
                    label=None if added_future_label else "ego future (all scenes)"
                )
                added_future_label = True

ax.set_aspect("equal")
ax.set_title("map_lanes + ego past/future (loop over scenes)")
ax.legend()
plt.show()
