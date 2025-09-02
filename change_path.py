#    *************************************************
#    PS： 这个程序保证每个文件夹下只有一个record文件，不然逻辑不对。 
#    *************************************************

# import os
# import shutil

# def distribute_files_to_folders(src_folder, prefix):
#     # 获取文件夹下的所有文件（排除子文件夹）
#     files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
#     for idx, file in enumerate(files, start=1):
#         # 子文件夹命名
#         folder_name = f"{prefix}_{idx}"
#         folder_path = os.path.join(src_folder, folder_name)
        
#         # 创建子文件夹（如果不存在）
#         os.makedirs(folder_path, exist_ok=True)
        
#         # 移动文件到子文件夹
#         src_file = os.path.join(src_folder, file)
#         dst_file = os.path.join(folder_path, file)
#         shutil.move(src_file, dst_file)
#         print(f"已移动: {file} -> {folder_name}")

# if __name__ == "__main__":
#     # 修改为你的文件夹路径, 保证一个文件夹只有一个.record文件。
#     source_directory = r"/home/pc/data_parser_train/task/0704"    # 包含多个.record文件的文件夹
#     distribute_files_to_folders(source_directory, prefix="0704")   # 前缀模板

import os
import shutil

def distribute_files_to_folders(src_folder, prefix):
    # 获取文件夹下的所有文件（排除子文件夹）
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    for idx, file in enumerate(files, start=1):
        # 子文件夹命名
        folder_name = f"{prefix}_{idx}"
        folder_path = os.path.join(src_folder, folder_name)
        
        # 创建子文件夹（如果不存在）
        os.makedirs(folder_path, exist_ok=True)
        
        # 移动文件到子文件夹
        src_file = os.path.join(src_folder, file)
        dst_file = os.path.join(folder_path, file)
        shutil.move(src_file, dst_file)
        print(f"已移动: {file} -> {folder_name}")

def process_all_subfolders(root_folder):
    # 遍历 root_folder 下的所有子文件夹
    for sub in os.listdir(root_folder):
        sub_path = os.path.join(root_folder, sub)
        if os.path.isdir(sub_path):
            print(f"\n处理子文件夹: {sub_path}")
            # 使用子文件夹名作为前缀
            distribute_files_to_folders(sub_path, prefix=sub)

if __name__ == "__main__":
    # 根目录，包含多个子文件夹，每个子文件夹下有多个 .record 文件
    root_directory = r"/home/pc/data_parser_train/task"
    process_all_subfolders(root_directory)
