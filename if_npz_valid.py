import numpy as np

# 读取文件
data = np.load("/home/WorkSpace/npz_save/sample_demo/ego_agent_past.npy")

# 查看基本信息
print("数据类型:", type(data))
print("数组形状:", data.shape)
print("数据类型(dtype):", data.dtype)

# 查看前几个元素
print("前5个数据:", data[:5])

# 输出三维数组: 
# 第一维 = 第几个人行横道（或其他对象）
# 第二维 = 这个对象包含多少个坐标点
# 第三维 = 单个点的 3 个坐标值（x, y, z）

###************** 好像只有一条斑马线数据 ***********