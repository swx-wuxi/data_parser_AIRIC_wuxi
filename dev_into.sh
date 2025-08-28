#!/bin/bash

# 容器名称
TARGET_CONTAINER_NAME="data_parser_5_12"

# 镜像名称
DOCKER_IMAGE="badaddressbin/data_parser:v0"

# 容器工作目录
WORKDIR="/home/WorkSpace"

# 宿主机项目挂载路径
HOST_PROJECT_PATH="/home/pc/data_parser_train"

# 容器内部的项目路径
CONTAINER_PROJECT_PATH="/home/WorkSpace/"

# 启用安全的 X11 本地访问
xhost +local:docker

# 函数：进入容器（优先 zsh，fallback bash）
enter_container() {
    echo -e "\033[1;32m进入容器 $TARGET_CONTAINER_NAME...\033[0m"
    sudo docker exec -it "$TARGET_CONTAINER_NAME" zsh 2>/dev/null || sudo docker exec -it "$TARGET_CONTAINER_NAME" bash
}

# 检查容器是否正在运行
running_container=$(sudo docker ps --filter "name=$TARGET_CONTAINER_NAME" --format "{{.ID}}")

if [ -n "$running_container" ]; then
    echo -e "\033[1;33m容器 $TARGET_CONTAINER_NAME 正在运行。\033[0m"
    enter_container
    exit 0
fi

# 检查容器是否已经存在（但未运行）
existing_container=$(sudo docker ps -a --filter "name=$TARGET_CONTAINER_NAME" --format "{{.ID}}")

if [ -n "$existing_container" ]; then
    echo -e "\033[1;34m容器 $TARGET_CONTAINER_NAME 存在，但未启动。\033[0m"
    sudo docker start "$TARGET_CONTAINER_NAME"
    enter_container
    exit 0
fi

# 容器不存在，准备创建并启动
echo -e "\033[1;36m容器 $TARGET_CONTAINER_NAME 不存在，创建并启动中...\033[0m"

sudo docker run -dit \
    --name="$TARGET_CONTAINER_NAME" \
    --runtime=nvidia \
    --gpus all \
    --privileged \
    --net=host \
    --ipc=host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$HOST_PROJECT_PATH":"$CONTAINER_PROJECT_PATH" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev:/dev \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev/shm:/dev/shm \
    -v /dev/bus/usb:/dev/bus/usb \
    --device=/dev/dri \
    --device=/dev/dri/renderD128 \
    --device=/dev/snd \
    --env="DISPLAY=$DISPLAY" \
    --group-add video \
    -w "$WORKDIR" \
    "$DOCKER_IMAGE" \
    /bin/bash

# 检查是否成功启动
if [ $? -ne 0 ]; then
    echo -e "\033[1;31m容器启动失败！\033[0m"
    exit 1
fi

echo -e "\033[1;32m容器 $TARGET_CONTAINER_NAME 启动成功。\033[0m"
enter_container

