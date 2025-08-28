# ./dev_into.sh
## 进入容器   

import os
import json
import pickle
import math
import glob
import numpy as np
import cv2  # 导入 OpenCV
import logging  # 新增日志模块
from collections import defaultdict
from cyber_record.record import Record
import bisect

from google.protobuf import text_format
import map_pb2

from collections import deque

# === 需要解析的 Topic 名称 ===
GLOBAL_POSE_TOPIC = "/apollo/localization/pose"
PERCEPTION_OBS_TOPIC = "/apollo/perception/obstacles"

TRAFFIC_LIGHT_TOPIC = "/rina/perception/tracked_traffic_lights"
ROUTE_INFO_TOPIC = "/rina/route_info"

# === 轨迹配置参数 ===
SCENE_DURATION = 20  # 场景固定时长20秒，采样频率10Hz
FRAME_TIMES = [3,13]  # 需要采样的帧时间点（秒）
HISTORY_DURATION = 2.0  # 历史轨迹时长 （20帧）
FUTURE_DURATION = 8.0   # 未来轨迹时长  （80帧）
TIME_TOLERANCE = 0.5 * 1e9  # 轨迹时间对齐容差
OB_TIME_TOLERANCE = 0.5 * 1e9  # object时间对齐容差
MAX_NEIGHBORS = 20      # 最大邻居数量

# === 地图配置参数 ===
MAP_PATH = "/home/WorkSpace/wuxi/base_map.bin"
MAP_QUERY_RADIUS = 60.0  # 地图查询范围60米
MAP_LANE_PARAMS = (40, 50)  # (最大数量, 每车道点数)
ROUTE_LANE_PARAMS = (10, 50)
CROSSWALK_PARAMS = (5, 30)

# === 障碍物类型映射（根据 proto 文件中的 ObstacleType 枚举）===
OBSTACLE_TYPE_MAP = {
    0: "UNKNOWN",          # ObstacleType.UNKNOWN
    1: "UNKNOWN_MOVABLE",  # ObstacleType.UNKNOWN_MOVABLE
    2: "UNKNOWN_UNMOVABLE",# ObstacleType.UNKNOWN_UNMOVABLE
    3: "CAR",              # ObstacleType.CAR
    4: "VAN",              # ObstacleType.VAN
    5: "TRUCK",            # ObstacleType.TRUCK
    6: "BUS",              # ObstacleType.BUS
    7: "CYCLIST",          # ObstacleType.CYCLIST
    8: "MOTORCYCLIST",     # ObstacleType.MOTORCYCLIST
    9: "TRICYCLIST",       # ObstacleType.TRICYCLIST
    10: "PEDESTRIAN",      # ObstacleType.PEDESTRIAN
    11: "CONE",            # ObstacleType.CONE
    12: "BICYCLE",         # ObstacleType.BICYCLE
    13: "SPLIT_VEHICLE",   # ObstacleType.SPLIT_VEHICLE
    14: "BARRIER",         # ObstacleType.BARRIER
    15: "WARNING_TRIANGLE",# ObstacleType.WARNING_TRIANGLE
    16: "ANIMAL",          # ObstacleType.ANIMAL
}

# === 障碍物类型映射（转换为训练类型） ===
TYPE_TO_CATEGORY = {
    "CAR": "vehicle",
    "VAN": "vehicle",
    "TRUCK": "vehicle",
    "BUS": "vehicle",
    "SPLIT_VEHICLE": "vehicle",
    "PEDESTRIAN": "pedestrian",
    "CYCLIST": "vehicle",
    "MOTORCYCLIST": "vehicle",
    "TRICYCLIST": "vehicle",
    "CONE": "other",
    "BICYCLE": "other",
    "BARRIER": "other",
    "WARNING_TRIANGLE": "other",
    "ANIMAL": "other",
    "UNKNOWN": "other",
    "UNKNOWN_MOVABLE": "other",
    "UNKNOWN_UNMOVABLE": "other"
}
def normalize_angle(a: float) -> float:
    # 归一化到 [-pi, pi]
    return (a + math.pi) % (2 * math.pi) - math.pi

def load_map_data(map_path):
    """加载高精地图数据"""
    try:
        with open(map_path, 'rb') as f:
            map_data = map_pb2.Map()
            map_data.ParseFromString(f.read())
        logging.info(f"成功加载高精地图，包含 {len(map_data.lane)} 条车道，{len(map_data.crosswalk)} 个人行横道")
        return map_data
    except Exception as e:
        logging.error(f"加载地图失败: {str(e)}")
        return None

def interpolate_points(points, target_num):
    """将点列表插值到指定数量"""
    if len(points) < 2 or target_num < 2:
        return points
    
    # 计算累积距离
    distances = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        distances.append(distances[-1] + np.hypot(dx, dy))
    
    # 生成等间距采样点
    new_points = []
    for t in np.linspace(0, distances[-1], target_num):
        idx = bisect.bisect_left(distances, t)
        if idx == 0:
            new_points.append(points[0])
        elif idx >= len(points):
            new_points.append(points[-1])
        else:
            ratio = (t - distances[idx-1]) / (distances[idx] - distances[idx-1])
            new_x = points[idx-1][0] + ratio * (points[idx][0] - points[idx-1][0])
            new_y = points[idx-1][1] + ratio * (points[idx][1] - points[idx-1][1])
            new_points.append([new_x, new_y])
    return new_points

def get_traffic_light_status(tl_list, current_time):
    """获取最近的交通灯状态"""
    if not tl_list:
        return defaultdict(lambda: 3)  # 默认unknown
    
    # 找到时间最近的状态
    times = [t[0] for t in tl_list]
    idx = bisect.bisect_left(times, current_time)
    idx = min(idx, len(tl_list)-1)
    return tl_list[idx][1]

def process_vector_map(ego_x, ego_y, ego_heading, 
                      hd_map, traffic_light_dict, current_nav_info=None):
    """处理矢量地图数据"""
    vector_map = {
        "map_lanes": [],
        "route_lanes": [],
        "map_crosswalks": []
    }
    
    COLOR_TO_INDEX = {
        0: 3,  # UNKNOWN_COLOR → unknown
        1: 2,  # RED → index2
        2: 0,  # GREEN → index0
        3: 1,  # YELLOW → index1
        4: 3   # BLACK → unknown
    }

    # === 处理普通车道 ===
    # logging.info("开始处理普通车道")
    lanes = []
    for lane in hd_map.lane:
        # 计算车道中心线到自车的最近距离
        # logging.info("======4a 计算车道中心线到自车的最近距离======")
        center_line = [[p.x, p.y] for p in lane.central_curve.segment[0].line_segment.point]
        if not center_line:
            continue
        
        # 转换到自车坐标系
        # logging.info("========4b 转换到自车坐标系=======")
        ego_coords = convert_to_ego_coordinate(np.array(center_line), ego_x, ego_y, ego_heading)
        distances = np.linalg.norm(ego_coords, axis=1)
        if np.min(distances) > MAP_QUERY_RADIUS:
            continue
        
        # 插值到固定点数
        # print("======4c.插值开始======")
        # 
        # ego_interp = convert_to_ego_coordinate(np.array(interpolated), ego_x, ego_y, ego_heading)

        # lane_data = []
        # for point in ego_interp:
        #     # 计算航向角（使用前后点差分）
        #     idx = min(interpolated.index(point), len(interpolated)-2)
        #     dx = interpolated[idx+1][0] - point[0]
        #     dy = interpolated[idx+1][1] - point[1]
        #     heading = math.atan2(dy, dx)
            
        #     # 获取交通灯状态
        #     if traffic_light_dict:
        #         # 获取第一个交通灯的颜色状态
        #         light_status = next(iter(traffic_light_dict.values()), 0)
        #     else:
        #         light_status = 0
        #     index = COLOR_TO_INDEX.get(light_status, 3)
        #     light_onehot = [0]*4
        #     light_onehot[index] = 1
            
        #     lane_data.append([
        #         point[0], point[1], heading,
        #         *light_onehot
        #     ])

        # （在插值之后，先转到自车系——这是你数值“几百万”的根因）
        # print("======4c.插值开始======")
        interpolated = interpolate_points(center_line, MAP_LANE_PARAMS[1])
        ego_interp = convert_to_ego_coordinate(np.array(interpolated), ego_x, ego_y, ego_heading)

        lane_data = []
        for i, point in enumerate(ego_interp):
            # 原来是：idx = min(interpolated.index(point), len(interpolated)-2)  # ❌ 会触发模糊布尔
            j = min(i, len(ego_interp) - 2)  # ✅ 直接用枚举出来的下标

            dx = ego_interp[j+1][0] - point[0]
            dy = ego_interp[j+1][1] - point[1]
            heading = math.atan2(dy, dx)

            # 交通灯 onehot（保持你原来的逻辑）
            if traffic_light_dict:
                light_status = next(iter(traffic_light_dict.values()), 0)
            else:
                light_status = 0
            index = COLOR_TO_INDEX.get(light_status, 3)
            light_onehot = [0]*4
            light_onehot[index] = 1

            # 写入“自车系”坐标，避免出现上百万的全局坐标
            lane_data.append([point[0], point[1], heading, *light_onehot])

        lanes.append((np.min(distances), lane_data))
    

    # 按距离排序并截断
    # logging.info("按距离排序并截断")
    lanes.sort(key=lambda x: x[0])
    for dist, lane in lanes[:MAP_LANE_PARAMS[0]]:
        vector_map["map_lanes"].append(lane)
    # 补零
    while len(vector_map["map_lanes"]) < MAP_LANE_PARAMS[0]:
        vector_map["map_lanes"].append([[0]*7]*MAP_LANE_PARAMS[1])
    
    # === 处理人行横道 ===
    # logging.info("处理人行横道")

    crosswalks = []
    for cw in hd_map.crosswalk:
        polygon = [[p.x, p.y] for p in cw.polygon.point]
        if len(polygon) < 3:
            continue
        # 转换到自车坐标系并检查距离
        ego_coords = convert_to_ego_coordinate(np.array(polygon), ego_x, ego_y, ego_heading)
        distances = np.linalg.norm(ego_coords, axis=1)
        if np.min(distances) > MAP_QUERY_RADIUS:
            continue
        # 取外边缘并插值
        interpolated = interpolate_points(polygon, CROSSWALK_PARAMS[1])
        crosswalks.append((np.min(distances), interpolated))
    
    # 排序截断
    crosswalks.sort(key=lambda x: x[0])
    for dist, cw in crosswalks[:CROSSWALK_PARAMS[0]]:
        # 转换到自车坐标系
        cw_ego = convert_to_ego_coordinate(np.array(cw), ego_x, ego_y, ego_heading)
        vector_map["map_crosswalks"].append([[p[0], p[1], 0] for p in cw_ego])
    # 补零
    while len(vector_map["map_crosswalks"]) < CROSSWALK_PARAMS[0]:
        vector_map["map_crosswalks"].append([[0]*3]*CROSSWALK_PARAMS[1])

    # === 处理导航路径车道 ===
    logging.info("=====4A.处理导航路径车道route_lanes=====")
    def find_current_lane(hd_map, ego_x, ego_y):
        """通过坐标查找当前所在车道"""
        min_dist = float('inf')
        current_lane = None
        for lane in hd_map.lane:
            # 快速筛选：检查车道边界框
            x_list = [p.x for p in lane.left_boundary.curve.segment[0].line_segment.point]
            y_list = [p.y for p in lane.left_boundary.curve.segment[0].line_segment.point]
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            # print(f"Ego位置: ({ego_x}, {ego_y})")        # 创新中心数据集提取出来的自车位置数据 
            # print(f"车道边界: x=({x_min}, {x_max}), y=({y_min}, {y_max})")  # hx地图数据
            # print(f"未匹配任何车道，最小距离为: {min_dist}")
            if not (x_min-20 <= ego_x <= x_max+20 and y_min-20 <= ego_y <= y_max+20):
                continue
            # 精确计算距离
        
            center_line = [[p.x, p.y] for p in lane.central_curve.segment[0].line_segment.point]
            ego_coords = convert_to_ego_coordinate(np.array(center_line), ego_x, ego_y, 0)
            distances = np.linalg.norm(ego_coords, axis=1)
            min_lane_dist = np.min(distances)
            if min_lane_dist < min_dist:
                min_dist = min_lane_dist
                current_lane = lane
        
        return current_lane

    def calculate_heading_change(current_lane, successor_lane):
        """计算车道末端到后继车道起始的航向变化"""
        try:
            # 当前车道末端方向
            current_points = current_lane.central_curve.segment[0].line_segment.point
            if len(current_points) < 2:
                return 0
            p1 = current_points[-2]
            p2 = current_points[-1]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            current_heading = math.atan2(dy, dx)
            # 后继车道起始方向
            successor_points = successor_lane.central_curve.segment[0].line_segment.point
            if len(successor_points) < 2:
                return 0
            p_start = successor_points[0]
            p_next = successor_points[1]
            dx_s = p_next.x - p_start.x
            dy_s = p_next.y - p_start.y
            successor_heading = math.atahd_mapn2(dy_s, dx_s)
            # 计算航向变化（规范化到[-pi, pi]）
            delta = successor_heading - current_heading
            return (delta + math.pi) % (2 * math.pi) - math.pi
        except:
            return 0

    # 步骤1：查找当前车道
    current_lane = find_current_lane(hd_map, ego_x, ego_y)
    # print(current_lane)
    if not current_lane:
        logging.warning("未找到当前所在导航车道！")
        return vector_map

    # 步骤2：收集相关车道（当前+同向邻居+后继）
    logging.info("=====4B.找到相关车道，开始搜集信息======")
    intersection_type = current_nav_info.intersection_type if current_nav_info else 0
    route_lane_set = set()
    current_lane_id = str(current_lane.id)
    route_lane_set.add(current_lane_id)
    # 添加当前车道的同向邻居
    for neighbor_id in current_lane.left_neighbor_forward_lane_id:
        neighbor = next((l for l in hd_map.lane if l.id == neighbor_id), None)
        if neighbor and neighbor.direction in [1,3]:
            route_lane_set.add(str(neighbor.id))
    for neighbor_id in current_lane.right_neighbor_forward_lane_id:
        neighbor = next((l for l in hd_map.lane if l.id == neighbor_id), None)
        if neighbor and neighbor.direction in [1,3]:
            route_lane_set.add(str(neighbor.id))
    # 处理当前车道的后继车道
    successors = []
    for successor_id in current_lane.successor_id:
        successor = next((l for l in hd_map.lane if l.id == successor_id), None)
        if not successor or successor.direction not in [1,3]:
            continue
        delta_heading = calculate_heading_change(current_lane, successor)
        valid = False
        if intersection_type == 2:  # LEFT
            valid = abs(delta_heading) > math.radians(30) and delta_heading > 0
        elif intersection_type == 3:  # RIGHT
            valid = abs(delta_heading) > math.radians(30) and delta_heading < 0
        elif intersection_type == 9:  # STRAIGHT
            valid = abs(delta_heading) < math.radians(30)
        else:
            valid = True
        
        if valid:
            successors.append(successor)

    # 添加后继及其同向邻居
    for successor in successors:
        route_lane_set.add(str(successor.id))
        for neighbor_id in successor.left_neighbor_forward_lane_id:
            neighbor = next((l for l in hd_map.lane if l.id == neighbor_id), None)
            if neighbor and neighbor.direction in [1,3]:
                route_lane_set.add(str(neighbor.id))
        for neighbor_id in successor.right_neighbor_forward_lane_id:
            neighbor = next((l for l in hd_map.lane if l.id == neighbor_id), None)
            if neighbor and neighbor.direction in [1,3]:
                route_lane_set.add(str(neighbor.id))
    # 步骤3：处理车道几何数据
    route_lanes = []
    # for lane_id in route_lane_set:
    #     lane = next((l for l in hd_map.lane if str(l.id) == lane_id), None)
    #     if not lane:
    #         continue
    #     # 提取中心线并转换坐标系
    #     center_line = [[p.x, p.y] for p in lane.central_curve.segment[0].line_segment.point]
    #     if not center_line:
    #         continue 
    #     # 距离计算
    #     ego_coords = convert_to_ego_coordinate(np.array(center_line), ego_x, ego_y, ego_heading)
    #     distances = np.linalg.norm(ego_coords, axis=1)
    #     if np.min(distances) > MAP_QUERY_RADIUS:
    #         continue
    #     # 插值处理
    #     interpolated = interpolate_points(center_line, ROUTE_LANE_PARAMS[1])
    #     lane_data = []
    #     for point in interpolated:
    #         idx = min(interpolated.index(point), len(interpolated)-2)
    #         dx = interpolated[idx+1][0] - point[0]
    #         dy = interpolated[idx+1][1] - point[1]
    #         heading = math.atan2(dy, dx)
    #         lane_data.append([point[0], point[1], heading])
    #     # 计算排序依据（取最近点距离）
    #     min_dist = np.min(distances)
    #     route_lanes.append((min_dist, lane_data))
    route_lanes = []
    for lane_id in route_lane_set:
        lane = next((l for l in hd_map.lane if str(l.id) == lane_id), None)
        if not lane:
            continue
        # 提取中心线并转换坐标系
        center_line = [[p.x, p.y] for p in lane.central_curve.segment[0].line_segment.point]
        if not center_line:
            continue 
        # 距离计算（用原始中心线做粗过滤）
        ego_coords = convert_to_ego_coordinate(np.array(center_line), ego_x, ego_y, ego_heading)
        distances = np.linalg.norm(ego_coords, axis=1)
        if np.min(distances) > MAP_QUERY_RADIUS:
            continue

        # 插值处理（仍在全局系）
        interpolated = interpolate_points(center_line, ROUTE_LANE_PARAMS[1])

        # ========= 仅此处新增：把插值后的点转到自车系 =========
        ego_interp = convert_to_ego_coordinate(np.array(interpolated), ego_x, ego_y, ego_heading)

        lane_data = []
        # ========= 仅此处改动：用枚举的下标，且在自车系下差分 =========
        for i, point in enumerate(ego_interp):
            j = min(i, len(ego_interp) - 2)
            dx = ego_interp[j+1][0] - point[0]
            dy = ego_interp[j+1][1] - point[1]
            heading = math.atan2(dy, dx)
            # 写入自车系坐标，避免全局系“上百万”
            lane_data.append([point[0], point[1], heading])

        # 计算排序依据（取最近点距离）——用自车系下的插值结果
        min_dist = float(np.min(np.linalg.norm(ego_interp, axis=1)))

        route_lanes.append((min_dist, lane_data))

    # 步骤4：排序和填充
    route_lanes.sort(key=lambda x: x[0])
    for dist, lane in route_lanes[:ROUTE_LANE_PARAMS[0]]:
        vector_map["route_lanes"].append(lane)
    # 补零
    while len(vector_map["route_lanes"]) < ROUTE_LANE_PARAMS[0]:
        vector_map["route_lanes"].append([[0]*3]*ROUTE_LANE_PARAMS[1])

    logging.info("=====4C返回vector map======")
    # print("Return vector map : %s",vector_map)
    return vector_map

def rotate_vector(vec, angle):
    """旋转二维向量到指定角度坐标系"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([
        vec[0] * cos_a - vec[1] * sin_a,
        vec[0] * sin_a + vec[1] * cos_a
    ])

# === 坐标系转换工具 ===
def convert_to_ego_coordinate(points, ego_x, ego_y, ego_heading):
    """
      将全局坐标转换为自车坐标系
      points: Nx2或Nx3的numpy数组
    """
    if len(points) == 0:
        return np.zeros((0, 2))
    # 平移
    translated = points - np.array([ego_x, ego_y]) 
    # 旋转
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)
    rotation_matrix = np.array([[cos_h, -sin_h],
                                [sin_h, cos_h]])
    rotated = np.dot(translated[:, :2], rotation_matrix)
    # 处理三维坐标
    if points.shape[1] == 3:
        return np.hstack([rotated, translated[:, 2:]])
    return rotated

def convert_from_ego_coordinate(points, ego_x, ego_y, ego_heading):
    """将自车坐标系坐标转换回全局坐标系"""
    if len(points) == 0:
        return np.zeros((0, 2))
    cos_h = np.cos(ego_heading)
    sin_h = np.sin(ego_heading)
    rotation_matrix = np.array([[cos_h, -sin_h],
                                [sin_h, cos_h]]).T  # 逆旋转
    
    rotated = np.dot(points[:, :2], rotation_matrix)
    translated = rotated + np.array([ego_x, ego_y])
    if points.shape[1] == 3:
        return np.hstack([translated, points[:, 2:]])
    return translated

def interpolate_trajectory(timestamps, positions, target_count=None):
    """线性插值缺失轨迹点，可指定目标点数"""
    if len(timestamps) == 0 or len(positions) == 0:
        return []
    # 生成目标时间戳
    if target_count is not None:
        start_time = timestamps[0]
        end_time = timestamps[-1]
        timestamps = np.linspace(start_time, end_time, target_count)
    interp_pos = []
    for t in timestamps:
        idx = bisect.bisect_left(timestamps, t)
        if idx == 0:
            interp_pos.append(positions[0])
        elif idx == len(timestamps):
            interp_pos.append(positions[-1])
        else:
            ratio = (t - timestamps[idx-1]) / (timestamps[idx] - timestamps[idx-1])
            interp = positions[idx-1] + ratio * (positions[idx] - positions[idx-1])
            interp_pos.append(interp)
    return interp_pos

# === 轨迹采样函数 ===
def sample_trajectory(timestamps, data_list, start_time, end_time, interval=0.1):
    """
    在指定时间范围内采样轨迹数据
    """
    if not timestamps or not data_list:
        logging.info("采样轨迹失败：定位数据时间戳或数据列表为空")
        return None

    # 检查时间窗口是否在数据范围内
    first_pose_time = timestamps[0]
    last_pose_time = timestamps[-1]
    if end_time < first_pose_time:
        logging.info(f"采样轨迹失败：时间窗口结束时间 {end_time/1e9:.3f}s 早于最早的定位数据时间 {first_pose_time/1e9:.3f}s")
        return None
    if start_time > last_pose_time:
        logging.info(f"采样轨迹失败：时间窗口起始时间 {start_time/1e9:.3f}s 晚于最晚的定位数据时间 {last_pose_time/1e9:.3f}s")
        return None

    sampled = []
    current_time = start_time
    failure_count = 0

    while current_time <= end_time:
        idx = bisect.bisect_left(timestamps, current_time)

        # 处理索引越界情况
        if idx >= len(timestamps) or idx < 0:
            logging.info(f"时间点 {current_time/1e9:.3f}s 超出定位数据范围")
            failure_count +=1
            current_time += interval * 1e9
            continue

        # 处理索引越界情况
        if idx < len(timestamps) and (timestamps[idx] - current_time) <= TIME_TOLERANCE:
            if idx < len(data_list):
                sampled.append(data_list[idx])
            else:
                logging.info("采样轨迹失败：索引超出数据列表范围")
                return None
        elif idx > 0 and (current_time - timestamps[idx-1]) <= TIME_TOLERANCE:
            if (idx-1) < len(data_list):
                sampled.append(data_list[idx-1])
            else:
                logging.info("采样轨迹失败：索引超出数据列表范围")
                return None
        else:
            # if idx >= len(timestamps) or idx <= 0:
            #     logging.info(f"采样轨迹失败：索引超出数据列表范围，当前搜寻id ：{idx} -> 当前数据长度：{len(timestamps)}")
            if (timestamps[idx] - current_time) > TIME_TOLERANCE:
                logging.info(f"采样轨迹失败：当前搜寻id:{idx}的时间不满足{TIME_TOLERANCE}的容忍范围，当前时间差为：{(timestamps[idx] - current_time)/1e9:.4f}s")
            return None
        current_time += interval * 1e9

    # 新增检查：无有效采样点
    if not sampled:
        time_window_str = f"{start_time/1e9:.3f}s → {end_time/1e9:.3f}s"
        data_range_str = f"{first_pose_time/1e9:.3f}s → {last_pose_time/1e9:.3f}s"
        logging.info(
            f"采样轨迹失败：时间窗口 {time_window_str} 内无有效数据\n"
            f"  可能原因：\n"
            f"  - 定位数据范围 {data_range_str} 不覆盖采样窗口\n"
            f"  - 数据间隔过大（当前采样间隔={interval}s）\n"
            f"  - 时间容差 {TIME_TOLERANCE/1e9:.3f}s 不足"
        )
        return None

    return sampled

# === 初始化日志配置 ===
def setup_logging(output_base_dir):
    log_file = os.path.join(output_base_dir, 'debug.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 文件处理器（输出到debug.log）
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 限制第三方库日志级别
    for lib in ['cyber_record', 'cyber_record.record']:
        logging.getLogger(lib).setLevel(logging.WARNING)

# === 查找 .record 文件 ===
# def traverse_files(base_dir):
#     """
#     遍历指定路径，查找所有的 .record 文件
#     """
#     return glob.glob(os.path.join(base_dir, "**", "orin-1", "gaea", "*.record"), recursive=True)
def traverse_files(base_dir: str):
    records = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            # 兼容以 .record 结尾，以及包含 ".record." 的分片文件
            if fname.endswith(".record") or ".record." in fname:
                full = os.path.join(root, fname)
                # 可选：过滤空文件
                try:
                    if os.path.getsize(full) > 0:
                        records.append(full)
                except OSError:
                    pass
    # 去重 + 排序（可选）
    records = sorted(set(records))
    return records

def find_nearest_with_tolerance(times, target, tolerance):
    """
    带容差的最近时间查找，返回索引或None
    """
    if not times:
        logging.info("时间列表为空，无法查找")
        return None
    idx = bisect.bisect_left(times, target)
    candidates = []
    if idx < len(times) and abs(times[idx] - target) <= tolerance:
        candidates.append(idx)
    if idx > 0 and abs(times[idx-1] - target) <= tolerance:
        candidates.append(idx-1)
    
    if not candidates:
        logging.info(f"在感知数据中未找到符合条件的时间点，target={target/1e9:.3f}s，容差={tolerance/1e9:.3f}s")
        return None
    return min(candidates, key=lambda i: abs(times[i] - target))

def process_neighbor_trajectory(obj, current_obs_time, history_timestamps, future_timestamps, 
                               obs_list, obs_times, global_pose_list, pose_times, 
                               ego_x, ego_y, ego_heading):
    """处理单个障碍物的完整轨迹"""
    # 获取自车状态函数
    def get_ego_state(t_ns):
        idx = bisect.bisect_left(pose_times, t_ns)
        idx = max(0, min(idx, len(global_pose_list)-1))
        pose = global_pose_list[idx][1]
        return (pose.pose.position.x, pose.pose.position.y, pose.pose.heading)

    # === 历史轨迹处理 ===
    past_states = []
    for t in history_timestamps:
        try:
            # 获取该时刻自车状态
            ego_x_t, ego_y_t, ego_h_t = get_ego_state(t)
            # 查找感知数据
            obs_idx = find_nearest_with_tolerance(obs_times, t, OB_TIME_TOLERANCE*2)
            found = None
            if obs_idx is not None and obs_idx < len(obs_list):
                found = next((o for o in obs_list[obs_idx][1].perception_obstacle 
                            if o.id == obj.id), None)
            if found:
                #***********************************原版代码*****************************************
                # # 坐标转换流程
                # # 历史自车坐标系 -> 全局 -> 当前自车坐标系

                # hist_local_pos = np.array([[found.position.x, found.position.y]])
                # hist_local_vel = np.array([found.velocity.x, found.velocity.y])
                # hist_local_acc = np.array([found.acceleration.x, found.acceleration.y])
                
                # # 转换到全局坐标系
                # global_pos = convert_from_ego_coordinate(
                #     hist_local_pos, ego_x_t, ego_y_t, ego_h_t)
                # global_vel = rotate_vector(hist_local_vel, ego_h_t)
                # global_acc = rotate_vector(hist_local_acc, ego_h_t)
                
                # # 转换到当前坐标系
                # current_local_pos = convert_to_ego_coordinate(
                #     global_pos, ego_x, ego_y, ego_heading)[0]
                # current_local_vel = rotate_vector(global_vel, -ego_heading)
                # current_local_acc = rotate_vector(global_acc, -ego_heading)
                
                # # 航向角处理（相对当前自车）
                # theta_global = found.theta + ego_h_t
                # theta_current = theta_global - ego_heading
                
                # past_states.append({
                #     'position': current_local_pos,
                #     'velocity': current_local_vel,
                #     'acceleration': current_local_acc,
                #     'theta': theta_current,
                #     'yaw_rate': 0.0,
                #     'timestamp': t
                # })
                # print(f"[DBG] t={t}: pos{current_local_pos.shape}, "
                #   f"vel{current_local_vel.shape}, acc{current_local_acc.shape}, "
                #   f"theta={type(theta_current)}, ts={type(t)}")
                #**************************************原版代码**************************************

                # === 世界(全局/ENU) -> 当前帧自车系 ===
                # 位置：先平移(减当前自车全局位置)，再绕(-ego_heading)旋转
                gx, gy = float(found.position.x), float(found.position.y)
                dx, dy = gx - ego_x, gy - ego_y
                cos_h, sin_h = math.cos(ego_heading), math.sin(ego_heading)
                current_local_pos = np.array([
                    dx * cos_h + dy * sin_h,     # x 前
                    -dx * sin_h + dy * cos_h     # y 左
                ], dtype=np.float64)

                # 速度/加速度：只需旋转到自车系（不需要平移）
                gvx, gvy = float(found.velocity.x), float(found.velocity.y)
                current_local_vel = np.array([
                    gvx * cos_h + gvy * sin_h,
                    -gvx * sin_h + gvy * cos_h
                ], dtype=np.float64)

                gax, gay = float(found.acceleration.x), float(found.acceleration.y)
                current_local_acc = np.array([
                    gax * cos_h + gay * sin_h,
                    -gax * sin_h + gay * cos_h
                ], dtype=np.float64)

                # 航向角：全局航向 - 当前自车航向（再归一化）
                theta_global = float(found.theta)
                theta_current = normalize_angle(theta_global - ego_heading)

                # # yaw_rate：先占位；推荐后面用相邻帧 theta 差分统一补（更稳）
                # yaw_rate_safe = float(getattr(found, "yaw_rate", 0.0) or 0.0)

                # （可选）调试打印：世界系->自车系的对照
                # print(f"[DBG] id={found.id} world=({gx:.2f},{gy:.2f}) ego=({current_local_pos[0]:.2f},{current_local_pos[1]:.2f})")

                past_states.append({
                    'position': current_local_pos,
                    'velocity': current_local_vel,
                    'acceleration': current_local_acc,
                    'theta': theta_current,
                    'yaw_rate': 0.0,   # 之后建议用差分覆盖
                    'timestamp': t
                })
                # print(f"[DBG] t={t}: pos{current_local_pos.shape}, "
                #   f"vel{current_local_vel.shape}, acc{current_local_acc.shape}, "
                #   f"theta={type(theta_current)}, ts={type(t)}")
            else:
                past_states.append(None)
        except Exception as e:
            logging.error(f"历史轨迹处理异常 @{t/1e9:.2f}s: {str(e)}")
            past_states.append(None)

    # print("=== Print First 3 past_states ===")
    # for i, s in enumerate(past_states[:3]):
    #   if s is None:
    #     print(f"[past#{i}] None")
    #   else:
    #     px, py = float(s['position'][0]), float(s['position'][1])
    #     vx, vy = float(s['velocity'][0]), float(s['velocity'][1])
    #     th = float(s['theta'])
    #     print(f"[past#{i}] t={s['timestamp']} pos=({px:.3f},{py:.3f}) vel=({vx:.3f},{vy:.3f}) theta={th:.3f} yaw_rate={s['yaw_rate']:.3f}")

    # 反向填充缺失的历史轨迹（使用后续数据推导）
    for i in range(len(past_states)-1, -1, -1):
        if past_states[i] is None:
            # 寻找后续第一个有效数据
            for j in range(i+1, len(past_states)):
                if past_states[j] is not None:
                    delta_t = (past_states[j]['timestamp'] - history_timestamps[i]) / 1e9
                    state_j = past_states[j]
                    
                    # 运动学方程反向推导
                    new_vel = state_j['velocity'] - state_j['acceleration'] * delta_t
                    new_pos = (
                        state_j['position'] 
                        - state_j['velocity'] * delta_t 
                        + 0.5 * state_j['acceleration'] * delta_t**2
                    )
                    
                    new_theta = state_j['theta'] - state_j['yaw_rate'] * delta_t
                    
                    past_states[i] = {
                        'position': new_pos,
                        'velocity': new_vel,
                        'acceleration': state_j['acceleration'],  # 假设加速度不变
                        'theta': new_theta,
                        # 'yaw_rate': state_j['yaw_rate'],
                        'yaw_rate':0.0,
                        'timestamp': history_timestamps[i]
                    }
                    break
            else:  # 没有找到后续有效数据
                past_states[i] = {
                    'position': np.zeros(2),
                    'velocity': np.zeros(2),
                    'acceleration': np.zeros(2),
                    'theta': 0.0,
                    'yaw_rate': 0.0,
                    'timestamp': history_timestamps[i]
                }

    # === 未来轨迹处理 ===
    future_states = []
    for t in future_timestamps:
        try:
            ego_x_t, ego_y_t, ego_h_t = get_ego_state(t)
            
            obs_idx = find_nearest_with_tolerance(obs_times, t, OB_TIME_TOLERANCE*2)
            found = None
            if obs_idx is not None and obs_idx < len(obs_list):
                found = next((o for o in obs_list[obs_idx][1].perception_obstacle 
                            if o.id == obj.id), None)
            
            if found:

                #**************************************原版代码************************************** 
                # # 坐标转换流程
                # fut_local_pos = np.array([[found.position.x, found.position.y]])
                # fut_local_vel = np.array([found.velocity.x, found.velocity.y])
                # fut_local_acc = np.array([found.acceleration.x, found.acceleration.y])
                
                # global_pos = convert_from_ego_coordinate(
                #     fut_local_pos, ego_x_t, ego_y_t, ego_h_t)
                # global_vel = rotate_vector(fut_local_vel, ego_h_t)
                # global_acc = rotate_vector(fut_local_acc, ego_h_t)
                
                # current_local_pos = convert_to_ego_coordinate(
                #     global_pos, ego_x, ego_y, ego_heading)[0]
                # current_local_vel = rotate_vector(global_vel, -ego_heading)
                # current_local_acc = rotate_vector(global_acc, -ego_heading)
                
                # theta_global = found.theta + ego_h_t
                # theta_current = theta_global - ego_heading
                
                # future_states.append({
                #     'position': current_local_pos,
                #     'velocity': current_local_vel,
                #     'acceleration': current_local_acc,
                #     'theta': theta_current,
                #     'yaw_rate': 0.0,
                #     'timestamp': t
                # })
                #**************************************原版代码**************************************

                gx, gy = float(found.position.x), float(found.position.y)
                dx, dy = gx - ego_x, gy - ego_y
                cos_h, sin_h = math.cos(ego_heading), math.sin(ego_heading)
                current_local_pos = np.array([
                    dx * cos_h + dy * sin_h,
                    -dx * sin_h + dy * cos_h
                ], dtype=np.float64)

                gvx, gvy = float(found.velocity.x), float(found.velocity.y)
                current_local_vel = np.array([
                    gvx * cos_h + gvy * sin_h,
                    -gvx * sin_h + gvy * cos_h
                ], dtype=np.float64)

                gax, gay = float(found.acceleration.x), float(found.acceleration.y)
                current_local_acc = np.array([
                    gax * cos_h + gay * sin_h,
                    -gax * sin_h + gay * cos_h
                ], dtype=np.float64)

                theta_current = normalize_angle(float(found.theta) - ego_heading)
                yaw_rate_safe = float(getattr(found, "yaw_rate", 0.0) or 0.0)

                future_states.append({
                    'position': current_local_pos,
                    'velocity': current_local_vel,
                    'acceleration': current_local_acc,
                    'theta': theta_current,
                    'yaw_rate': 0.0,
                    'timestamp': t
                })
                
            else:
                future_states.append(None)
        except Exception as e:
            logging.error(f"未来轨迹处理异常 @{t/1e9:.2f}s: {str(e)}")
            future_states.append(None)
        
    # 正向填充缺失的未来轨迹（使用先前数据推导）
    for i in range(len(future_states)):
        if future_states[i] is None:
            # 寻找前一个有效数据
            for j in range(i-1, -1, -1):
                if future_states[j] is not None:
                    delta_t = (future_timestamps[i] - future_states[j]['timestamp']) / 1e9
                    state_j = future_states[j]
                    
                    # 运动学方程正向推导
                    new_vel = state_j['velocity'] + state_j['acceleration'] * delta_t
                    new_pos = (
                        state_j['position'] 
                        + state_j['velocity'] * delta_t 
                        + 0.5 * state_j['acceleration'] * delta_t**2
                    )
                    new_theta = state_j['theta'] + state_j['yaw_rate'] * delta_t
                    
                    future_states[i] = {
                        'position': new_pos,
                        'velocity': new_vel,
                        'acceleration': state_j['acceleration'],  # 假设加速度不变
                        'theta': new_theta,
                        # 'yaw_rate': state_j['yaw_rate'],
                        'yaw_rate':0.0,
                        'timestamp': future_timestamps[i]
                    }
                    break
            else:  # 没有找到先前有效数据
                future_states[i] = {
                    'position': np.zeros(2),
                    'velocity': np.zeros(2),
                    'acceleration': np.zeros(2),
                    'theta': 0.0,
                    'yaw_rate': 0.0,
                    'timestamp': future_timestamps[i]
                }

    return past_states, future_states

# === 单个 record 处理函数 ===
def process_record(record_path, output_base_dir):
    """
    处理单个 record 文件并生成输出
    """
    #logging.info(f"\n===== 处理文件：{record_path} =====")
    record = Record(record_path)

    # 加载高精地图（提前加载）
    hd_map = load_map_data(MAP_PATH)
    if not hd_map:
        logging.error("无法加载高精地图")
        return

    # 收集各消息并按时间排序（使用纳秒时间戳）
    
    all_data = []
    traffic_light_list = []  # 存储交通灯信息
    route_info_list = []
    for topic, msg, t_ns in record.read_messages():
        # print(f"[DEBUG] topic: '{topic}'")
        if topic == GLOBAL_POSE_TOPIC:
            # logging.info("=====Subscribe position info=======")
            all_data.append(('global_pose', msg, t_ns))
        elif topic == PERCEPTION_OBS_TOPIC:
            # logging.info("=====Subscribe perception info=======")
            all_data.append(('obs', msg, t_ns))
        elif topic == ROUTE_INFO_TOPIC:  # 导航信息处理
            route_info_list.append((t_ns, msg))
        elif topic == TRAFFIC_LIGHT_TOPIC:  # 交通灯处理
            light_status = {}
            for light in msg.traffic_light:
                light_status[light.id] = light.color
            traffic_light_list.append((t_ns, light_status))

    # print(f"导航路径车道ID示例：{route_lane_ids[:5]}")

    all_data.sort(key=lambda x: x[2])  # 按纳秒排序

    # 按类型存储数据
    global_pose_list = [item for item in all_data if item[0] == 'global_pose']
    obs_list = [item for item in all_data if item[0] == 'obs']

    ############# Debug #############
    # print(len(global_pose_list),"##")
    # print(len(obs_list),"##")

    # print("global_pose_list属性:")
    # for item in global_pose_list:
    #     print(item)

    ############# Debug #############

    if not global_pose_list:
        logging.info("未找到GlobalPose数据，跳过该文件")
        return
    
    # 检查数据有效性
    logging.info("==========开始读取感知数据=========")
    if not obs_list:
        logging.info("无感知数据，跳过文件处理")
        return
    if len(obs_list)<50:
        logging.info("感知数据少于50，数据不正常，跳过文件处理")
        return
    
    record_start_ns = all_data[0][2]
    record_end_ns = all_data[-1][2]
    record_duration = (record_end_ns - record_start_ns) / 1e9
    logging.info(f"数据包时间范围：{record_start_ns/1e9:.3f}s - {record_end_ns/1e9:.3f}s (总时长：{record_duration:.2f}秒)")

    #############################################
    # 调试代码：检测定位数据是否异常
    #############################################
    logging.info("\n===== 调试信息：定位数据分析 =====")
    logging.info(f"定位数据总量：{len(global_pose_list)} 条")
    
    if len(global_pose_list) < 2:
        logging.warning("定位数据不足，无法分析间隔")
        return

    # # 时间间隔分析
    # pose_times = [x[2] for x in global_pose_list]
    # intervals = []
    # abnormal_intervals = []
    
    # for i in range(1, len(pose_times)):
    #     delta_ns = pose_times[i] - pose_times[i-1]
    #     delta_sec = delta_ns / 1e9
    #     intervals.append(delta_sec)
        
    #     if delta_sec > 0.11:  # 考虑10%的误差容忍
    #         abnormal_intervals.append((
    #             pose_times[i-1]/1e9,  # 间隔开始时间(s)
    #             pose_times[i]/1e9,    # 间隔结束时间(s)
    #             delta_sec            # 间隔时长(s)
    #         ))

    # # 统计信息
    # if intervals:
    #     logging.info(f"时间间隔统计：min={min(intervals):.3f}s | "
    #                  f"max={max(intervals):.3f}s | "
    #                  f"avg={np.mean(intervals):.3f}s")
    #     logging.info(f"超过0.1s的异常间隔数量：{len(abnormal_intervals)}")
        
    #     # 输出前10个异常间隔
    #     if abnormal_intervals:
    #         logging.info("\n[ 异常间隔列表 ] 格式：开始时间 → 结束时间 | 时长")
    #         for start, end, delta in abnormal_intervals[:10]:
    #             logging.info(f"{start:7.3f}s → {end:7.3f}s | {delta:.3f}s")
    #         if len(abnormal_intervals) > 10:
    #             logging.info(f"...（共{len(abnormal_intervals)}条，仅显示前10条）")
                
    #     # 数据量合理性检查
    #     total_time = (pose_times[-1] - pose_times[0]) / 1e9
    #     expected_count = int(total_time * 100)  # 100Hz的期望
    #     ratio = len(global_pose_list) / expected_count if expected_count>0 else 0
    #     logging.info(f"\n数据量合理性评估：")
    #     logging.info(f"  数据时间范围：{pose_times[0]/1e9:.1f}s → {pose_times[-1]/1e9:.1f}s (总时长：{total_time:.1f}s)")
    #     logging.info(f"  实际数据量：{len(global_pose_list)} | 理论数据量(100Hz)：{expected_count}")
    #     logging.info(f"  数据完整率：{ratio*100:.1f}%")
    #     if ratio < 0.9:
    #         logging.warning("⚠️ 数据完整率低于90%，可能存在问题！")
    # else:
    #     logging.info("无有效间隔数据")


    # logging.info("\n=== 调试信息：定位数据重复检测 ===")
    # identical_periods = []
    # current_start_ns = None
    # last_unique_data = None
    
    # # 遍历全局定位数据
    # for i in range(len(global_pose_list)):
    #     msg = global_pose_list[i][1]
    #     t_ns = global_pose_list[i][2]
        
    #     # 提取关键字段（保留6位小数避免浮点误差）
    #     current_data = (
    #         # round(msg.position_enu.x, 6),
    #         # round(msg.position_enu.y, 6),
    #         # round(msg.linear_velocity.x, 6),
    #         # round(msg.linear_velocity.y, 6)
    #         msg.position_enu.x,
    #         msg.position_enu.y,
    #         msg.linear_velocity.x,
    #         msg.linear_velocity.y
    #     )
        
    #     # 比较数据变化
    #     if last_unique_data != current_data:
    #         if current_start_ns is not None:
    #             # 记录时间段（排除小于0.5秒的情况）
    #             duration_ns = t_ns - current_start_ns
    #             if duration_ns >= 0.5e9:
    #                 identical_periods.append((
    #                     current_start_ns / 1e9,
    #                     global_pose_list[i-1][2] / 1e9,
    #                     duration_ns / 1e9,
    #                     last_unique_data
    #                 ))
    #         current_start_ns = t_ns
    #         last_unique_data = current_data
    
    # # 输出重复时间段信息
    # for start_t, end_t, duration, data in identical_periods:
    #     logging.info(f"定位数据相同时段：{start_t:.1f}s -> {end_t:.1f}s "
    #           f"（持续{duration:.1f}s）\n"
    #           f"  固定值：位置=({data[0]}, {data[1]}) "
    #           f"速度=({data[2]}, {data[3]}) ")

    #############################################
    # 调试代码：感知数据时间间隔分析
    #############################################
    logging.info("\n=== 调试信息：感知数据分析 ===")
    if not obs_list:
        logging.info("无感知数据")
        return
    # 计算感知时间范围（转换为秒）
    first_obs_sec = obs_list[0][2] / 1e9
    last_obs_sec = obs_list[-1][2] / 1e9
    logging.info(f"感知时间范围：{first_obs_sec:.3f}s ~ {last_obs_sec:.3f}s")
    # 生成全间隔列表（带时间戳）
    interval_records = []
    for i in range(1, len(obs_list)):
        prev_time = obs_list[i-1][2]
        curr_time = obs_list[i][2]
        delta_sec = (curr_time - prev_time) / 1e9
        interval_records.append((
            i,  # 间隔序号
            prev_time / 1e9,  # 起始时间(s)
            curr_time / 1e9,  # 结束时间(s)
            delta_sec  # 间隔时长(s)
        ))
    # 输出统计信息
    if interval_records:
        deltas = [x[3] for x in interval_records]
        logging.info(f"感知帧数：{len(obs_list)} 总间隔数：{len(interval_records)}")
        logging.info(f"间隔统计：min={min(deltas):.3f}s | max={max(deltas):.3f}s | avg={np.mean(deltas):.3f}s")
        # 打印所有间隔（每行一个间隔）
        # logging.info("\n[ 全部感知间隔 ] 格式：序号 | 时间段 | 时长 | 状态")
        # for idx, start, end, delta in interval_records:
        #     status = "[!异常!]" if delta > 1.0 else ""
        #     logging.info(f"#{idx:03d} | {start:7.3f}s → {end:7.3f}s | {delta:5.3f}s {status}")
        # 输出异常间隔汇总
        abnormal = [ (r[0], r[3]) for r in interval_records if r[3] > 1.0 ]
        if abnormal:
            logging.info("\n异常间隔汇总：")
            for pos, t in abnormal:
                logging.info(f"  间隔#{pos} 持续{t:.3f}s")
        else:
            logging.info("\n无超过1秒的异常间隔")
    else:
        logging.info("仅1帧感知数据，无间隔可计算")


    # 生成时间戳列表
    pose_times = [x[2] for x in global_pose_list]
    obs_times = [x[2] for x in obs_list]

    # 划分场景窗口
    scene_windows = []
    current_start = pose_times[0]
    while current_start < pose_times[-1]:
        scene_end = current_start + SCENE_DURATION*1e9
        # 允许最后一个场景最小19秒
        if scene_end > pose_times[-1]:
            if (pose_times[-1] - current_start) >= 19e9:
                scene_end = pose_times[-1]
            else:
                break
        scene_windows.append((current_start, scene_end))
        current_start = scene_end

    logging.info("场景划分结束")
    
    # 处理每个场景
    for scene_idx, (scene_start, scene_end) in enumerate(scene_windows):
        # 收集场景数据
        scenario_data = []

        # 处理每个采样帧
        for frame_offset in FRAME_TIMES:
            logging.info(f"**********开始处理第{scene_idx}个场景中的第{frame_offset}s采样帧***********")
            target_time = scene_start + int(frame_offset * 1e9)
            
            # 查找最近的感知数据
            obs_idx = find_nearest_with_tolerance(obs_times, target_time, OB_TIME_TOLERANCE)
            if obs_idx is None:
                # logging.info(f"在时间 {target_time/1e9:.3f}s 处未找到感知数据")
                continue

            # 获取当前帧信息（增加索引越界保护）
            if obs_idx >= len(obs_list) or obs_idx < 0:
                logging.info(f"obs_idx {obs_idx} 越界，列表长度 {len(obs_list)}")
                continue

            current_obs = obs_list[obs_idx]
            # print("DEBUG type:", type(current_obs[1]), "value:", current_obs[1])

            if len(current_obs) < 3 or current_obs[0] != 'obs':
                logging.info(f"Invalid obs entry: {current_obs}")
                continue
            if not hasattr(current_obs[1], 'perception_obstacle'):
                logging.info("current_obs message missing perception_obstacle")
                continue

            current_pose_idx = bisect.bisect_left(pose_times, current_obs[2])
            current_pose_idx = min(current_pose_idx, len(global_pose_list)-1)
            current_pose_idx = max(current_pose_idx, 0)
            # logging.info(f"当前帧定位数据索引 {current_pose_idx}/{len(global_pose_list)}")
            current_pose = global_pose_list[current_pose_idx][1]
            
            # 坐标系参数

            ##########################  IMPORTANT DEBUG CODE #############################

            # logging.info("current_pose 类型: %s", type(current_pose))
            # logging.info("current_pose 内容:\n%s", current_pose)

            ##########################  IMPORTANT DEBUG CODE #############################

            ego_x = current_pose.pose.position.x
            ego_y = current_pose.pose.position.y
            ego_heading = current_pose.pose.heading
         
            # === 1. 自车历史轨迹 ===
            logging.info("================1. 采样自车历史轨迹===============")
            history_start = current_obs[2] - int(HISTORY_DURATION * 1e9)
            history_samples = sample_trajectory(pose_times, global_pose_list, 
                                                history_start, current_obs[2], 0.1)
            
            # print(len(history_samples),"---")
            if not history_samples or len(history_samples) < 21:
                logging.info("历史采样轨迹数据不足21个点，跳过该帧")
                continue
            
            ego_past = []
            for sample in history_samples:
                pose = sample[1]   # global_pose_list
                # 转换到当前坐标系
                global_pos = np.array([[pose.pose.position.x, pose.pose.position.y]])
                local_pos = convert_to_ego_coordinate(global_pos, ego_x, ego_y, ego_heading)[0]
                ego_past.append([
                    local_pos[0], local_pos[1],
                    pose.pose.heading - ego_heading,
                    pose.pose.linear_velocity.x,
                    pose.pose.linear_velocity.y,
                    pose.pose.linear_acceleration.x,
                    pose.pose.linear_acceleration.y
                ])
           
            # === 2. 自车未来轨迹 ===
            logging.info("===============2. 开始采样自车未来轨迹==============")
            future_start = current_obs[2] + int(0.1 * 1e9)
            future_end = current_obs[2] + int(FUTURE_DURATION * 1e9)
            future_samples = sample_trajectory(pose_times, global_pose_list,
                                              future_start, future_end, 0.1)
            
            if not future_samples or len(future_samples) < 70:
                logging.info("未来采样轨迹数据不足70个点，跳过该帧")
                continue
                
            ego_future = []
            for sample in future_samples:
                pose = sample[1]
                global_pos = np.array([[pose.pose.position.x, pose.pose.position.y]])
                local_pos = convert_to_ego_coordinate(global_pos, ego_x, ego_y, ego_heading)[0]
                ego_future.append([
                    local_pos[0],
                    local_pos[1],
                    pose.pose.heading - ego_heading
                ])
            # print("Future trajecotry info: %s",ego_future)
            # === 3. 邻居车辆处理 ===
            # ************************核心修改点***************************
            logging.info("==================3. 开始感知车辆处理===============")
            current_obstacles = {obj.id: obj for obj in current_obs[1].perception_obstacle}
            neighbors = []

            # 时间戳生成
            history_time_offsets = np.linspace(0, HISTORY_DURATION, int(HISTORY_DURATION/0.1)+1)[::-1]
            future_time_offsets = np.arange(0.1, FUTURE_DURATION+0.1, 0.1)
            history_timestamps = [current_obs[2] - int(t*1e9) for t in history_time_offsets]
            future_timestamps = [current_obs[2] + int(t*1e9) for t in future_time_offsets]
            
            # logging.info("====采样感知车辆历史、未来轨迹====")
            for obj in current_obstacles.values():
                # 获取完整轨迹
                past_states, future_states = process_neighbor_trajectory(
                    obj, current_obs[2], 
                    history_timestamps, future_timestamps,
                    obs_list, obs_times, 
                    global_pose_list, pose_times,
                    ego_x, ego_y, ego_heading
                )

                # 类型编码
                #logging.info("类型编码")
                obj_type = OBSTACLE_TYPE_MAP.get(obj.type, "UNKNOWN")
                category = TYPE_TO_CATEGORY.get(obj_type, "other")
                one_hot = [1,0,0] if category == "vehicle" else \
                         [0,1,0] if category == "pedestrian" else [0,0,1]
                
                # 组装轨迹数据
                #logging.info("组装轨迹数据")
                neighbor_past = []
                for state in past_states:
                    neighbor_past.append([
                        state['position'][0], state['position'][1],
                        state['theta'],
                        state['velocity'][0], state['velocity'][1],
                        state['yaw_rate'],
                        obj.width, obj.length,
                        *one_hot
                    ])

                neighbor_future = [[
                    s['position'][0], s['position'][1],
                    s['theta']
                ] for s in future_states]
                
                neighbors.append({
                    "past": neighbor_past,
                    "future": neighbor_future
                })     
            
            # 按距离排序并截断
            #logging.info("按距离排序并截断")
            if neighbors:
                neighbors.sort(
                    key=lambda x: np.linalg.norm(x["past"][-1][:2]) if x["past"] else float('inf'),
                    reverse=True
                )
                neighbors = neighbors[:MAX_NEIGHBORS]
            else:
                logging.info("当前帧无有效感知数据")

            # === 4. 高精地图处理 ===
            logging.info("===============4. 高精地图处理==============")
            current_time = current_obs[2]
            traffic_light_dict = get_traffic_light_status(traffic_light_list, current_time)

            route_info_idx = bisect.bisect_left([t for t, _ in route_info_list], current_time)
            current_route_info = None
            if route_info_list:
                if route_info_idx == 0:
                    current_route_info = route_info_list[0][1]
                elif route_info_idx >= len(route_info_list):
                    current_route_info = route_info_list[-1][1]
                else:
                    before = route_info_list[route_info_idx-1][1]
                    after = route_info_list[route_info_idx][1]
                    current_route_info = after if (route_info_list[route_info_idx][0] - current_time) < \
                        (current_time - route_info_list[route_info_idx-1][0]) else before
            
            # logging.info("获取当前导航信息中的第一个NavigationInfo")
            # 获取当前导航信息中的第一个NavigationInfo
            current_nav_info = None
            if current_route_info and current_route_info.navigation_info:
                current_nav_info = current_route_info.navigation_info[0]
            
            vector_map = process_vector_map(ego_x, ego_y, ego_heading, hd_map, 
                                           traffic_light_dict, current_nav_info)

            # === 组装最终数据 ===
            logging.info("==============5.组装最终数据=============")
            frame_data = {
                "ego_agent_past": [ego_past],
                "ego_agent_future": [ego_future],
                "neighbor_agents_past": [n["past"] for n in neighbors],
                "neighbor_agents_future": [n["future"] for n in neighbors],
                "vector_map": vector_map
            }
            scenario_data.append(frame_data)

        # 从路径中获取时间戳部分
        
        parts = record_path.split("/")
        # print(parts)
        # if len(parts) < 8:
        #     logging.info(f"无效文件路径格式：{record_path}")
        #     continue
        task_folder = parts[-2]  # 获取任务文件夹名称
        timestamp = parts[-1]  # 获取时间戳部分
        timestamp = timestamp.split('.')[0]    # 获取文件名中前面那一串小数部分。

        # 保存场景数据
        # if len(scenario_data) == 2:
        if scenario_data: 
            output_dir = os.path.join(output_base_dir, task_folder, timestamp)
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, f"scenario_{scene_idx}.json"), 'w') as f:
                json.dump(scenario_data, f, indent=2)
            logging.info(f"保存文件：scenario_{scene_idx}.json")

    # # 生成可视化视频(尚未开发)
    # try:
    #     visualize_fast(
    #         os.path.join(output_dir, "scenario.json"),
    #         os.path.join(output_dir, "scenario.mp4")
    #     )
    #     print(f"🎬 可视化视频已保存: {os.path.join(output_dir, 'scenario.mp4')}")
    # except Exception as e:
    #     print(f"❌ 可视化失败: {e}")

# === 主入口  ===
# 代码块参考： 576行临车处理流程  762行是自车（核心流程）
# 地图处理过程：133行开始
if __name__ == "__main__":
    base_dir = "/home/WorkSpace/task"
    output_base_dir = "/home/WorkSpace/output"
    os.makedirs(output_base_dir, exist_ok=True)
    
    record_files = traverse_files(base_dir)
    
    setup_logging(output_base_dir)
    
    # 记录系统信息
    logging.info("===== 数据处理任务开始 =====")
    logging.info(f"输入目录：{base_dir}")
    logging.info(f"输出目录：{output_base_dir}")
    logging.info(f"找到 {len(record_files)} 个 record 文件")

    for idx, path in enumerate(record_files, 1):
        try:
            logging.info(f"\n🔧 正在处理第 {idx}/{len(record_files)} 个文件：{path}")
            process_record(path, output_base_dir)
        except Exception as e:
            logging.info(f"处理文件 {path} 时发生错误：{str(e)}")

    logging.info("===== 数据处理任务完成 =====")

