from cyber.python.cyber_py3 import record

REC = "/home/WorkSpace/task/task_0724_1/20250724152934.record"
TOPIC = "/apollo/perception/obstacles"

rr = record.RecordReader(REC)
cnt_msgs = 0
cnt_obs  = 0

for topic, msg, t in rr.read_messages(topics=[TOPIC]):
    cnt_msgs += 1
    # 标准命名（最常见）
    if hasattr(msg, "perception_obstacle"):
        cnt_obs += len(msg.perception_obstacle)
    # 兼容其它可能命名
    elif hasattr(msg, "perception_obstacles"):
        cnt_obs += len(msg.perception_obstacles)
    elif hasattr(msg, "obstacles"):
        cnt_obs += len(msg.obstacles)

print("messages on topic:", cnt_msgs)
print("total obstacles in all msgs:", cnt_obs)
