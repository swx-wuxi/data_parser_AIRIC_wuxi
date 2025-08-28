import numpy as np
from pathlib import Path

# 输出目录
out_path = Path("/home/WorkSpace/output_npz/sample_0.npz")
# 需要的文件名以及占位 shape
FILES = {
    "lanes.npy": (1, 4),
    "route_lanes.npy": (1, 4),
    "route.npy": (1, 4),
    "crosswalks.npy": (1, 4),
    "ego_agent_past.npy": (1, 10, 3),
    "ego_agent_future.npy": (1, 10, 3),
    "ego_past.npy": (1, 10, 3),
    "ego_future.npy": (1, 10, 3),
    "neighbor_agents_past.npy": (1, 5, 10, 3),
    "neighbor_agents_future.npy": (1, 5, 10, 3),
    "neighbors_past.npy": (1, 5, 10, 3),
    "neighbors_future.npy": (1, 5, 10, 3),
    "traffic_light.npy": (1, 4),
    "acc_classification.npy": (1,),
    "lane_change.npy": (1,),
    "ego_lane_flag.npy": (1,),
    "neighbour_lane.npy": (1,),
    "ego_v_a.npy": (1, 2),
    "token.npy": (1,),
    "instruction.npy": (1,),
    "iter.npy": (1,),
    "map_name.npy": (1,)
}

# 用零填充所有字段
arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in FILES.items()}

# 保存到 npz
np.savez(out_path, **arrays)

print("空 npz 文件已生成:", out_path)
