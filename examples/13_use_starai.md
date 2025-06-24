# 如何搭建具身智能Lerobot violin和cello机械臂并完成自定义抓取任务

[LeRobot](https://github.com/huggingface/lerobot/tree/main) 致力于为真实世界的机器人提供 PyTorch 中的模型、数据集和工具。其目标是降低机器人学的入门门槛，使每个人都能通过共享数据集和预训练模型进行贡献和受益。LeRobot 集成了经过验证的前沿方法，专注于模仿学习和强化学习。它提供了一套预训练模型、包含人类收集的示范数据集和仿真环境，使用户无需进行机器人组装即可开始使用。未来几周，计划在当前最具成本效益和性能的机器人上增强对真实世界机器人的支持。



##  Starai系列特点

1. **开源 & 便于二次开发**
   本系列舵机由[华馨京科技](https://fashionrobo.com/)提供，是一套开源、便于二次开发的6+1自由度机器臂解决方案。
2. **支持 LeRobot 平台集成**
   专为与 [LeRobot 平台](https://github.com/huggingface/lerobot) 集成而设计。该平台提供 PyTorch 模型、数据集与工具，面向现实机器人任务的模仿学习（包括数据采集、仿真、训练与部署）。
3. **丰富的学习资源**TODO
   提供全面的开源学习资源，帮助用户快速上手并开发机器人应用。



## 特点内容：

- 自由度：拥有6+1自由度，抵达位置更广泛。
- 悬停按钮功能支持：悬停按钮可让LeaderArm随时悬停在任意角度。



# 硬件规格参数

|                      | Violin(Leader)                                               | Viola(Follower)                                              | Cello(Follower)                                              |
| -------------------- | ------------------------------------------------------------ | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **推荐电源**         | DC 12V 10 A                                                  | DC 12V 10 A                                                  | DC 12V 25A                                                   |
| **电机构成**         | RX8-U50H-M x2 <br>RA8-U25H-M x5                              | RX8-U50H-M x2 <br/>RA8-U25H-M x5                             | RX18-U100H-M x3 <br/>RX8-U50H-M x4                           |
| **悬停**             | 通过手柄按钮实现                                             | 跟随Leader时可悬停                                           | 跟随Leader时可悬停                                           |
| **推荐工作温度范围** | 0 °C ～ 60 °C                                                | 0 °C ～ 60 °C                                                | 0 °C ～ 60 °C                                                |
| **关节范围**         | joint0:±135°<br>joint1:±90°<br/>joint2:±90°<br/>joint3:±135°<br/>joint4:±90°<br/>joint5:±135°<br/> | joint0:±135°<br/>joint1:±90°<br/>joint2:±90°<br/>joint3:±135°<br/>joint4:±90°<br/>joint5:±135°<br/> | joint0:±135°<br/>joint1:±90°<br/>joint2:±90°<br/>joint3:±135°<br/>joint4:±90°<br/>joint5:±135°<br/> |
| **夹爪行程**         | 0~70mm                                                       | 0~70mm                                                       | 0~70mm                                                       |

TODO:机械臂示意图

有关舵机更多资讯，请访问以下链接。

[RA8-U25H-M](https://fashionrobo.com/actuator-u25/23396/)

[RX18-U100H-M](https://fashionrobo.com/actuator-u100/22853/)

[RX8-U50H-M](https://fashionrobo.com/actuator-u50/136/)







# 初始系统环境

For Ubuntu X86:

- Ubuntu 22.04
- CUDA 12+
- Python 3.13
- Troch 2.6





# 安装 LeRobot

需要根据你的 CUDA 版本安装 pytorch 和 torchvision 等环境。

1. 安装 Miniconda： 对于 Jetson：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh
source ~/.bashrc
```

或者，对于 X86 Ubuntu 22.04：

```bash
mkdir -p ~/miniconda3
cd miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

2.创建并激活一个新的 conda 环境用于 lerobot

```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

3.克隆 Lerobot 仓库：

```bash

```


4.使用 miniconda 时，在环境中安装 ffmpeg：

```bash
conda install ffmpeg -c conda-forge
```
这通常会为你的平台安装使用 libsvtav1 编码器编译的 ffmpeg 7.X。如果不支持 libsvtav1（可以通过 ffmpeg -encoders 查看支持的编码器），你可以：

- 【适用于所有平台】显式安装 ffmpeg 7.X：

```bash
conda install ffmpeg=7.1.1 -c conda-forge
```

5.安装带有 fashionstar 电机依赖的 LeRobot：

```bash
cd ~/lerobot && pip install -e ".[starai]"
```
重新安装

```bash
cd ~/lerobot && pip install -e ".[starai]"

pip install --only-binary :all: pynput
```

6.检查 Pytorch 和 Torchvision

由于通过 pip 安装 lerobot 环境时会卸载原有的 Pytorch 和 Torchvision 并安装 CPU 版本，因此需要在 Python 中进行检查。

```python
import torch
print(torch.cuda.is_available())
```

如果输出结果为 False，需要根据[官网教程](https://pytorch.org/index.html)重新安装 Pytorch 和 Torchvision。




## 手臂端口设置

在终端输入以下指令来找到两个机械臂对应的端口号：

```bash
python lerobot/scripts/find_motors_bus_port.py
```

例如：

1. 识别Leader时端口的示例输出（例如，在 Mac 上为 `/dev/tty.usbmodem575E0031751`，或在 Linux 上可能为 `/dev/ttyUSB0`） 
2. 识别Reader时端口的示例输出（例如，在 Mac 上为 `/dev/tty.usbmodem575E0032081`，或在 Linux 上可能为 `/dev/ttyUSB1`）





注意：如果识别不到ttyUSB0串口信息。

列出所有usb口。

```sh
lsusb
```

![image-20241230112928879-1749511998299-1](./../media/starai/image-20241230112928879-1749511998299-1.png)

识别成功，查看ttyusb的信息

```sh
sudo dmesg | grep ttyUSB
```

![image-20241230113058856](./../media/starai/image-20241230113058856-1749512093309-2.png)

最后一行显示断连，因为brltty在占用该USB设备号，移除掉就可以了

```sh
sudo apt remove brltty
```

![image-20241230113211143](./../media/starai/image-20241230113211143-1749512102599-4.png)

最后，赋予权限

```sh
sudo chmod 666 /dev/ttyUSB0
```



## 遥操作机械臂（不包括摄像头）

您已准备好遥操作您的机器人！运行以下简单脚本：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

程序启动后，悬停按钮依旧生效。



## 添加摄像头

在插入您的两个 USB 摄像头后，运行以下脚本以检查摄像头的端口号，切记摄像头不能插在USB Hub上，要直接插在设备上，USB Hub速率太慢会导致读不到图像数据。

```bash
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images_from_opencv_cameras
```

终端将打印出以下信息。

```markdown
Mac or X86 Ubuntu detected. Finding available camera indices through scanning all indices from 0 to 60
[...]
Camera found at index 2
Camera found at index 4
[...]
Connecting cameras
OpenCVCamera(2, fps=30.0, width=640, height=480, color_mode=rgb)
OpenCVCamera(4, fps=30.0, width=640, height=480, color_mode=rgb)
Saving images to outputs/images_from_opencv_cameras
Frame: 0000 Latency (ms): 39.52
[...]
Frame: 0046 Latency (ms): 40.07
Images have been saved to outputs/images_from_opencv_cameras
```

您可以在 `outputs/images_from_opencv_cameras` 目录中找到每个摄像头拍摄的图片，并确认不同位置摄像头对应的端口索引信息。然后，完成 `lerobot/lerobot/common/robot_devices/robots/configs.py` 文件中摄像头参数的对齐。

```
@RobotConfig.register_subclass("starai")
@dataclass
class StaraiRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/starai"

    cameras: dict[str, CameraConfig] = field(
        default_factory=*lambda*: {
            "laptop": OpenCVCameraConfig(
                camera_index=2,             ##### UPDATE HEARE
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=0,             ##### UPDATE HEARE
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

​    mock: bool = False
  
```

然后，您将能够在遥操作时在计算机上显示摄像头：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=teleoperate \
  --control.display_data=true
```

## 数据集制作采集

```python
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=${HF_USER}/starai_test \
  --control.tags='["starai01","tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=20 \
  --control.display_data=true \
  --control.push_to_hub=false
```

- 如果你希望将数据保存在本地（`--control.push_to_hub=false`），请将 `--control.repo_id=${HF_USER}/starai_test` 替换为一个自定义的本地文件夹名称，例如 `--control.repo_id=starai_123/starai_test`。数据将存储在系统主目录下的 `~/.cache/huggingface/lerobot`。
- 如果你通过 `--control.push_to_hub=true` 将数据集上传到了 Hugging Face Hub，可以通过 [在线可视化你的数据集](https://huggingface.co/spaces/lerobot/visualize_dataset)，只需复制粘贴你的 repo id。
- 注意：你可以通过添加 `--control.resume=true` 来继续录制。如果你还没有上传数据集，还需要添加 `--control.local_files_only=true`。
- 在回合记录过程中任何时候按下右箭头 -> 可提前停止并进入重置状态。重置过程中同样，可提前停止并进入下一个回合记录。
- 在录制或重置到早期阶段时，随时按左箭头 <- 可提前停止当前剧集，并重新录制。
- 在录制过程中随时按 ESCAPE ESC 可提前结束会话，直接进入视频编码和数据集上传。
- 一旦你熟悉了数据记录，你就可以创建一个更大的数据集进行训练。一个不错的起始任务是在不同的位置抓取物体并将其放入箱子中。我们建议至少记录 50 个场景，每个位置 10 个场景。保持相机固定，并在整个录制过程中保持一致的抓取行为。同时确保你正在操作的物体在相机视野中可见。一个很好的经验法则是，你应该仅通过查看相机图像就能完成这项任务。
- 在接下来的章节中，你将训练你的神经网络。在实现可靠的抓取性能后，你可以在数据收集过程中引入更多变化，例如增加抓取位置、不同的抓取技巧以及改变相机位置。 
- 避免快速添加过多变化，因为这可能会阻碍您的结果。
- 在 Linux 上，如果在数据记录期间左右箭头键和 Esc 键没有效果，请确保您已设置 $DISPLAY 环境变量。参见 [pynput 限制](https://pynput.readthedocs.io/en/latest/limitations.html#linux)。



## 可视化数据集

数据集保存在本地，并且数据采集时运行超参数为 `--control.push_to_hub=false` ，您也可以使用以下命令在本地进行可视化：

```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id starai_123/starai_test \
```

这里的`starai_123`为采集数据时候自定义的repo_id名。



## 重播一个回合

```
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=starai_123/starai_test \
  --control.episode=0
  --control.local_files_only=true
```



## 训练

要训练一个控制您机器人策略，使用 `python lerobot/scripts/train.py` 脚本。需要一些参数。以下是一个示例命令：

```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=starai_123/starai_test \
  --policy.type=act \
  --output_dir=outputs/train/act_starai_test \
  --job_name=act_starai_test \
  --policy.device=cuda \
  --wandb.enable=false
```

如果你想训练本地数据集，repo_id与采集数据的repo_id对齐即可。训练应该需要几个小时。您将可以在 `outputs/train/act_starai_test/checkpoints` 中找到训练结果的权重文件。

要从某个检查点恢复训练，下面是一个从 `act_starai_test` 策略的最后一个检查点恢复训练的示例命令：

```bash
python lerobot/scripts/train.py \
  --config_path=outputs/train/act_starai_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 评估

您可以使用 [`lerobot/scripts/control_robot.py`](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/control_robot.py) 中的 `record` 功能，但需要将策略训练结果权重作为输入。例如，运行以下命令记录 10 个评估回合：

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=starai \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Grasp a lego block and put it in the bin." \
  --control.repo_id=starai_123/eval_act_starai_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/train/act_so101_test/checkpoints/last/pretrained_model
```

