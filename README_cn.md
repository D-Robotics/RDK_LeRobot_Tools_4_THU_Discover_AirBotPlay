[English](./README.md) | 简体中文

# 在RDK板卡中利用HuggingFace的LeRobot框架使用求之AIRBOT机械臂

在RDK设备上适配LeRobot框架的ACT Policy端到端具身VA算法, 并使用这套框架成功挑战叠衣服的案例, 给了我们很大的信心, 我们深刻地体会到了LeRobot让具身智能的开发更简单. 

本文以求之AIRBOT机械臂为例, 详细阐述了如何在LeRobot框架中新增一款Robot, 接入HuggingFace开源生态, 并借助LeRobot框架在您的Robot快速实现遥操、数采, 将您的机器人数据集保存为标准的LeRobot格式，并由此借助LeRobot框架来进行算法训练, 快速复现各种Policy算法.

![](imgs/LeRobot_ReRun.jpeg)
![](imgs/train_a_policy.jpeg)

本文所有的源代码文件和README文档托管在

GitHub: https://github.com/D-Robotics/RDK_LeRobot_Tools_4_THU_Discover_AirBotPlay



**往期参考:**

GitHub: https://github.com/D-Robotics/rdk_LeRobot_tools

NodeHub: https://developer.d-robotics.cc/nodehubdetail/1918884279126081537


## 一、求之 AIRBOT 简介

![](imgs/introduction2AIRBOT1.png)

求之科技怀揣“让机器人走进千家万户”的愿景, 坚持利用智能机器人技术赋能产业升级、推动社会进步. 为加速具身智能技术发展与应用落地, 求之科技精心打造AIRBOT系列产品, 为高校与科研机构提供具身智能与机器人的全方位解决方案。

AIRBOT 产品网站: https://airbots.online/zh

AIRBOT 产品资料: https://airbots.online/zh/dowload

AIRBOT 产品文档: https://docs.airbots.online/zh/airbot-play/quick-start/overview/


## 二、HuggingFace LeRobot框架简介

![](imgs/lerobot-logo-light.png)

🤗 LeRobot 旨在为现实世界的机器人技术提供基于 PyTorch 的模型、数据集和工具。其目标是降低机器人领域的入门门槛，让每个人都能参与贡献，并从共享的数据集和预训练模型中受益。

🤗 LeRobot 包含了当前最先进的方法，这些方法已被证明可以迁移到真实世界任务中，重点集中在模仿学习和强化学习方向。

🤗 LeRobot 已经提供了多个预训练模型、包含人类操作演示的数据集，以及仿真环境，让你无需实际组装机器人即可开始研究和开发。在接下来的几周内，我们计划不断增加对市面上最经济实惠且性能出色的机器人设备的支持。

🤗 所有预训练模型和数据集都可以在这个 Hugging Face 社区页面上找到：`huggingface.co/lerobot`

GitHub: https://github.com/huggingface/lerobot

HuggingFace: https://huggingface.co/lerobot

## 三、如何为 Hugging Face 的 LeRobot框架添加一款新的Robot？

### 3.1 了解LeRobot管理和使用各种机器人类(对象)的方式

#### 机器人的类
在LeRobot的项目中, `lerobot/common/robot_devices/robots/` 文件夹下有很多预设的机器人配置, 这里存放着机器人的类(Class), 在LeRobot的各种功能实现中, 会通过`lerobot/common/robot_devices/robots/utils.py`模块中实现的`make_robot`等方法来实例化某一种机器人类, 返回机器人对象, 从而对您的机器进行各种操作. 

```bash
lerobot/common/robot_devices/robots
├── configs.py
├── dynamixel_calibration.py
├── feetech_calibration.py
├── lekiwi_remote.py
├── manipulator.py
├── mobile_manipulator.py
├── stretch.py
├── thu_airbot.py
└── utils.py
```

#### 机器人配置类

每一种机器人的初始化配置放置在 `lerobot/common/robot_devices/robots/configs.py` 文件中. 这里在后文会详细介绍.

```bash
@RobotConfig.register_subclass("THU_AIRBOT")
@dataclass
class THU_AIRBOTConfig(ManipulatorRobotConfig):
    leader_arms: ...
    follower_arms: ...
    cameras: ...
```

### 流程分析
![](imgs/lerobot_control_loop.png)

遥操、数采和Policy算法的实机验证都是通过`lerobot/scripts/control_robot.py` 脚本实现。

指定`--control.tyep=teleoperate` 参数, 从而运行 `teleoperate()` 函数, 以 `teleoperate=True`参数调用`control_loop()` 函数, 实现遥操.

指定`--control.tyep=record` 参数, 从而运行 `record()` 函数, 以 `teleoperate=True, dataset is not None`参数调用`control_loop()` 函数, 实现数采.

指定`--control.tyep=record` 参数, 从而运行 `record()` 函数, 以 `teleoperate=True, policy is not None`参数调用`control_loop()` 函数, 实现数采.


### 3.2 梳理您自己的机器人的SDK

求之AIRBOT提供Python API, 我们就可以很方便的通过Python API来获取机械臂的状态, 给机械臂下发控制命令.

目前, 如果要通过LeRobot项目来控制您自己的机器人, 需要梳理得到以下SDK的API. 以下以求之AIRBOT为例, 需要梳理以下API.

#### (1) 连接

导入您自己机器人的包, 这取决于您使用的机器人的SDK.  

然后基于您自己机器人的SDK, 可以做一些简单的设置, 一般主臂(Leader)使用重力补偿模式, 从臂(Follower)使用被控制的模式.

```python
# Reference Code (AIRBOT Example)
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile

# Leader
leader = AIRBOTPlay(url=cfg.ip, port=cfg.port)
leader.connect()
leader.switch_mode(RobotMode.GRAVITY_COMP)

# Folllower
follower = AIRBOTPlay(url=cfg.ip, port=cfg.port)
follower.connect()
follower.switch_mode(RobotMode.PLANNING_POS)
follower.set_speed_profile(SpeedProfile.FAST)
```


#### (2) 获取机械臂状态

使用您自己机器人的SDK, 来获取主臂的状态, 并弄清楚返回的数据类型, 一般是Python的列表(list), 或者Numpy数组(numpy.array), 或者其他的数据格式, 一般都可以转化为PyTorch的张量(torch.tensor), 这是LeRobot所需要的.

```python
# Reference Code (AIRBOT Example)

# Get observation.state
pos = leader.get_joint_pos()  # Returns: list[float]
eef = leader.get_eef_pos()    # Returns: list[float]

# Convert to torch.tensor
pos = np.array(pos + eef, dtype=np.float32)  # Returns: numpy.array
pos_tensor = torch.from_numpy(pos).float()   # Returns: torch.tensor
```

#### (3) 限位

限位一般会在您自己机器人的SDK手册中提供, 为了防止机器人进入死区或者不合理的状态, 求之AIRBOT提供每个自由度的最大和最小值, 我们可以在给从臂(Follower)下发位置前进行数值的限制.

```python
# Reference Code (AIRBOT Example)

# Get Max and Min limits
limit_max = [2.089, 0.181, 3.161, 3.012, 1.859, 3.017]
limit_min = [-3.151, -2.963, -0.094, -3.012, -1.859, -3.017]

# Clip
for i in range(6):
    pos[i] = pos[i] if pos[i] < self.limit_max[i] else self.limit_max[i]
    pos[i] = pos[i] if pos[i] > self.limit_min[i] else self.limit_min[i]

```

#### (4) 下发机械臂状态

通过您自己机器人的SDK手册提供的方式下发机械臂状态.

```python
# Reference Code (AIRBOT Example)
follower.servo_joint_pos(pos)
follower.servo_eef_pos(eef_pos)
```


#### (5) 断开

通过您自己机器人的SDK手册提供的方式安全断开机械臂.

```python
# Reference Code (AIRBOT Example)
leader.set_speed_profile(SpeedProfile.DEFAULT)
follower.set_speed_profile(SpeedProfile.DEFAULT)
```

### 3.3 新增机器人的配置类

新增机器人的配置类实现在`lerobot/common/robot_devices/robots/configs.py`文件, 在文件内进行程序设计即可, 这里的书写方式与Python的类(Class)是一样的.

这里对每一个臂的参数新增了`THU_AIRBOT_Play_SigleARM_Config`类, 用于管理每一个臂的IP和端口号.

Camera参数使用的是LeRobot原版的Camera的配置, 这意味着所有和Camera相关的使用和原来LeRobot的使用是一致的. 

```python
class THU_AIRBOT_Play_SigleARM_Config:
    ip: str
    port: int
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

@RobotConfig.register_subclass("THU_AIRBOT")
@dataclass
class THU_AIRBOTConfig(ManipulatorRobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left_leader": THU_AIRBOT_Play_SigleARM_Config(
                ip=airbot_ip,
                port=50053,
            ),
            "right_leader": THU_AIRBOT_Play_SigleARM_Config(
                ip=airbot_ip,
                port=50054,
            ),
        }
    )
    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left_follower": THU_AIRBOT_Play_SigleARM_Config(
                ip=airbot_ip,
                port=50051,
            ),
            "right_follower": THU_AIRBOT_Play_SigleARM_Config(
                ip=airbot_ip,
                port=50052,
            ),
        }
    )
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "laptop": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "phone": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    mock: bool = False
```


### 3.4 新增机器人的类

新增`lerobot/common/robot_devices/robots/thu_airbot.py`文件, 在这里实现求之AIRBOT的类. 为了方便, 您也可以从已经实现的机器人类中来修改. 

我们新增自己的机器人类, 需要实现以下方法:

```bash
robot.__init__()
robot.connect() 
robot.disconnect()
robot.capture_observation()
robot.teleop_step()
robot.send_action()
```

#### (1) THU_AIRBOT.__init__() 方法的实现

这里包含您自己机器人的初始化, 需要着重关注的是如何从`config`中读取我们想要的配置项目, 这里逐级去索引即可.

关于摄像头的初始化, 我们使用的是`LeRobot`原版的`make_cameras_from_configs()`方法, 传入的参数也是`config`配置对象.

```python
def __init__(self, config: THU_AIRBOTConfig):
    """
    Expected keys in config:
        - ip, port, video_port for the remote connection.
        - calibration_dir, leader_arms, follower_arms, max_relative_target, etc.
    """
    self.robot_type = config.type
    self.config = config
    self.is_connected = False
    # Your Robot
    self.leader_arms = {}
    for key, cfg in self.config.leader_arms.items():
        self.leader_arms[key] = AIRBOTPlay(url=cfg.ip,port=cfg.port)
    self.follower_arms = {}
    for key, cfg in self.config.follower_arms.items():
        self.follower_arms[key] = AIRBOTPlay(url=cfg.ip,port=cfg.port)
    # Origin LeRobot Cameras
    self.cameras = make_cameras_from_configs(self.config.cameras)
    ## pos limit
    self.limit_max = [2.089, 0.181, 3.161, 3.012, 1.859, 3.017]
    self.limit_min = [-3.151, -2.963, -0.094, -3.012, -1.859, -3.017]
```


#### (2) THU_AIRBOT.connect() 方法的实现

这里逐个去调用各个组件的连接方式即可, 特别的, Camera我们使用了异步读取, 以减少OpenCV的读帧延迟. 

```python
def connect(self):
    # Leader
    for name in self.leader_arms:
        self.leader_arms[name].connect()
        self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
    # Follower
    for name in self.follower_arms:
        self.follower_arms[name].connect()
        self.follower_arms[name].switch_mode(RobotMode.PLANNING_POS)
        self.follower_arms[name].set_speed_profile(SpeedProfile.FAST)
    # cameras
    for cam_name, cam in self.cameras.items():
        cam.connect()
        cam.async_read()
    
    self.safe_begin()
        
    self.is_connected = True

```


#### (3) THU_AIRBOT.disconnect() 方法的实现

disconnect() 方法由这个类的析构函数调用, 我们可以写一些善后的逻辑, 以避免机器人行为异常, 避免潜在的危险.

```python
def disconnect(self):
    if not self.is_connected:
        raise RobotDeviceNotConnectedError("Not connected.")
    # Leader
    for name in self.leader_arms:
        self.leader_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
    # Follower 
    for name in self.follower_arms:
        self.follower_arms[name].switch_mode(RobotMode.GRAVITY_COMP)
    # cameras
    for cam_name, cam in self.cameras.items():
        cam.disconnect()
    self.is_connected = False
```

#### (4) THU_AIRBOT.capture_observation() 方法的实现

这里调用您自己的机器人的方式去获取数据, 特别的, 摄像头的数据直接获取图像帧的引用即可, 如果不放心可以使用`copy.deepcopy()`方法对图像帧进行深拷贝.

```python
    def capture_observation(self) -> dict:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

        # Get obeservation.states from Follower Arms
        follower_arm_positions = []
        for name in self.follower_arms:
            pos = self.follower_arms[name].get_joint_pos()
            eef = self.follower_arms[name].get_eef_pos()
            pos = np.array(pos + eef, dtype=np.float32)
            pos_tensor = torch.from_numpy(pos).float()
            follower_arm_positions.extend(pos_tensor.tolist())
        follower_arm_positions = torch.tensor(follower_arm_positions, dtype=torch.float32)
        obs_dict = {"observation.state": follower_arm_positions}

        # Get obeservation.images from Camera(s)
        for cam_name, cam in self.cameras.items():
            frame = cam.color_image
            if frame is None:
                frame = np.zeros((cam.height, cam.width, cam.channels), dtype=np.uint8)
            obs_dict[f"observation.images.{cam_name}"] = torch.from_numpy(frame)

        return obs_dict
```



#### (5) THU_AIRBOT.teleop_step() 方法的实现

这里是遥操的主循环, 主要的逻辑是实现一次主臂各关节角度下发从臂的运动控制, 然后从从臂获取关节角度作为state的关节数据. 最后将获取到的数据按照LeRobot的规范打包为词典, 然后返回.

```python
def teleop_step(self, record_data: bool = False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if not self.is_connected:
        raise RobotDeviceNotConnectedError("THU_AIRBOT is not connected. Run `connect()` first.")
    
    # Get States from Leader Arms 
    leader_arm_positions = []
    for name in self.leader_arms:
        pos = self.leader_arms[name].get_joint_pos()
        for i in range(6):
            pos[i] = pos[i] if pos[i] < self.limit_max[i] else self.limit_max[i]
            pos[i] = pos[i] if pos[i] > self.limit_min[i] else self.limit_min[i]
        eef = self.leader_arms[name].get_eef_pos()
        pos = np.array(pos + eef, dtype=np.float32)
        pos_tensor = torch.from_numpy(pos).float()
        leader_arm_positions.extend(pos_tensor.tolist())

    # Send Action to Follower Arms
    self.send_action(leader_arm_positions)
    time.sleep(0.01)

    # For Record
    if not record_data:
        return

    # Get obeservation.states from Follower Arms
    # Get obeservation.images from Camera(s)
    obs_dict = self.capture_observation()

    # Returns
    arm_state_tensor = torch.tensor(leader_arm_positions, dtype=torch.float32)
    action_dict = {"action": arm_state_tensor}

    return obs_dict, action_dict
```


#### (6) THU_AIRBOT.send_action() 方法的实现

这里调用您自己的机器人的方式去下发数据即可, 特别的, 这里可以加入限位的逻辑.

```python
def send_action(self, action: torch.Tensor) -> torch.Tensor:
    if not self.is_connected:
        raise RobotDeviceNotConnectedError("Not connected. Run `connect()` first.")

    slice_begin = 0
    slice_end = 7
    for follower in self.follower_arms:
        self.follower_arms[follower].servo_joint_pos(action[slice_begin:slice_end-1])
        self.follower_arms[follower].servo_eef_pos(action[slice_end-1:slice_end])
        slice_begin += 7
        slice_end += 7

    return action
```

### 3.5 导入您的机器人配置

您可以复制一份`lerobot/scripts/control_robot.py`文件, 例如`control_my_robot.py`, 替换`robot = make_robot_from_config(cfg.robot)`语句, 然后直接使用此配置来生成机器人.

```python
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    # robot = make_robot_from_config(cfg.robot)
    from lerobot.common.robot_devices.robots.thu_airbot import THU_AIRBOT
    robot = THU_AIRBOT(cfg.robot)
    ...
```



## 四、使用您自己实现的机器人类(Class)来遥操

```bash
python lerobot/scripts/control_my_robot.py \
  --robot.type=THU_AIRBOT \
  --robot.cameras='{}' \
  --control.type=teleoperate
```


## 五、使用您自己实现的机器人类来数采

这里除了调用自己的脚本, 其他参数与原版的LeRobot一致. 这意味着您可以进行可视化的数采, 同时对生成的数据集使用LeRobot的方式进行可视化.

```bash
python lerobot/scripts/control_my_robot_thu.py  \
 --robot.type=THU_AIRBOT   \
 --control.type=record   \
 --control.fps=30   \
 --control.single_task="Fold shirt."  \
 --control.warmup_time_s=10   \
 --control.episode_time_s=150   \
 --control.reset_time_s=50000   \
 --control.num_episodes=100   \
 --control.push_to_hub=false   \
 --control.display_data=true   \
 --control.repo_id=USER/so100_test \
 --control.root=datasets/debug_test \
 --control.resume=false
```

可视化效果参考

![](imgs/LeRobot_ReRun.jpeg)


## 六、使用您自己实现的机器人类来实机验证Policy算法

使用record方法使推理和记录评估数据集同步进行.

```bash
python lerobot/scripts/control_my_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Cloth Fold" \
  --control.repo_id=USER/eval_act_so100_0418_test1 \ 
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=180 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=outputs/act_so100_resnet152_0418_test1/pretrained_model
```

![](imgs/train_a_policy.jpeg)