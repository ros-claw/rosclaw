# ROSClaw MCP Hub 详细设计

> Universal Embodied MCP Gateway - 连接任意Agent与物理世界
> 融合来源: awesome-mcp-servers, OpenClaw MCP集成, awesome-mcp-clients

---

## 一、架构定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ROSClaw MCP Hub Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  External Agents (Layer 6+)                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ Claude Code │  │ OpenClaw    │  │ WorkBuddy   │  │ Custom Agent    │    │
│  │             │  │             │  │             │  │                 │    │
│  │ JSON-RPC    │  │ JSON-RPC    │  │ JSON-RPC    │  │ JSON-RPC        │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘    │
│         │                │                │                   │             │
│         └────────────────┴────────────────┴───────────────────┘             │
│                              ↓ MCP Protocol                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         MCP Hub Gateway                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Protocol Layer                                                │ │ │
│  │  │  ├── Transport: stdio / HTTP / WebSocket / SSE                 │ │ │
│  │  │  ├── Serialization: JSON-RPC 2.0                               │ │ │
│  │  │  └── Authentication: API Key / JWT / mTLS                      │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                              ↓                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Tool Router                                                     │ │ │
│  │  │  ├── Static Routing (预定义)                                     │ │ │
│  │  │  ├── Dynamic Routing (运行时注册)                                 │ │ │
│  │  │  └── Load Balancing (多机器人场景)                                │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  │                              ↓                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Safety Middleware                                               │ │ │
│  │  │  ├── Command Validation (Digital Twin Firewall)                  │ │ │
│  │  │  ├── Rate Limiting (频率限制)                                     │ │ │
│  │  │  └── Audit Logging (审计日志)                                     │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  ROSClaw OS Kernel (Layers 1-4)                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Tool Registry                                                        │ │
│  │  ├── Motion Tools (移动、机械臂、关节)                                  │ │
│  │  ├── Perception Tools (相机、激光雷达、IMU)                             │ │
│  │  ├── Planning Tools (导航、避障、路径规划)                               │ │
│  │  ├── Manipulation Tools (抓取、操作、装配)                               │ │
│  │  └── System Tools (状态、诊断、配置)                                    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                              ↓                                               │
│  Physical Runtime (Layer 5)                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ ROS 2       │  │ VLA Engine  │  │ Hardware    │                         │
│  │ DDS         │  │ OpenVLA/π0  │  │ Drivers     │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、协议层设计

### 2.1 传输层支持

```python
# src/rosclaw/mcp/transports.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class MCPTransport(ABC):
    """MCP传输层抽象"""

    @abstractmethod
    async def connect(self): ...

    @abstractmethod
    async def send(self, message: dict): ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[dict]: ...

    @abstractmethod
    async def close(self): ...

class StdioTransport(MCPTransport):
    """标准输入输出传输 - Claude Code默认"""

    async def connect(self):
        self.reader = asyncio.StreamReader()
        self.writer = sys.stdout

    async def send(self, message: dict):
        self.writer.write(json.dumps(message).encode() + b'\n')
        self.writer.flush()

class WebSocketTransport(MCPTransport):
    """WebSocket传输 - 远程连接"""

    async def connect(self, uri: str):
        self.websocket = await websockets.connect(uri)

    async def send(self, message: dict):
        await self.websocket.send(json.dumps(message))

    async def receive(self):
        async for message in self.websocket:
            yield json.loads(message)

class HTTPTransport(MCPTransport):
    """HTTP传输 - 无状态请求"""

    async def send(self, message: dict):
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=message) as resp:
                return await resp.json()
```

### 2.2 认证与安全

```python
# src/rosclaw/mcp/auth.py
class MCPAuthenticator:
    """MCP连接认证器"""

    SUPPORTED_METHODS = ['api_key', 'jwt', 'mtls', 'none']

    def __init__(self, method: str, config: dict):
        self.method = method
        self.config = config

    async def authenticate(self, request: dict) -> AuthResult:
        """认证请求"""
        if self.method == 'api_key':
            return self._verify_api_key(request.get('headers', {}).get('x-api-key'))

        elif self.method == 'jwt':
            return self._verify_jwt(request.get('headers', {}).get('authorization'))

        elif self.method == 'mtls':
            return self._verify_client_cert(request.get('tls', {}))

        return AuthResult(authenticated=True, permissions=['read', 'write'])

    def _verify_api_key(self, api_key: str) -> AuthResult:
        """API Key验证"""
        if api_key not in self.config['allowed_keys']:
            return AuthResult(authenticated=False, error="Invalid API key")

        permissions = self.config['key_permissions'].get(api_key, ['read'])
        return AuthResult(authenticated=True, permissions=permissions)
```

---

## 三、Tool 设计

### 3.1 复合工具结构

参考 `better-godot-mcp` 的18个复合工具设计，按功能域组织：

```python
# src/rosclaw/mcp/tools/__init__.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rosclaw-hub")

# ============================================
# Motion Domain (运动控制域)
# ============================================
@mcp.tool()
def motion_move_base(linear_x: float, angular_z: float) -> dict:
    """
    移动机器人底盘

    Args:
        linear_x: 线速度 (m/s)
        angular_z: 角速度 (rad/s)
    """
    ...

@mcp.tool()
def motion_move_arm(joint_positions: list, speed: float = 0.5) -> dict:
    """
    移动机械臂到目标关节位置

    Args:
        joint_positions: 目标关节角度列表 (rad)
        speed: 运动速度 (0.0-1.0)
    """
    ...

@mcp.tool()
def motion_move_cartesian(pose: dict, frame: str = "world") -> dict:
    """
    笛卡尔空间移动

    Args:
        pose: 目标位姿 {position: {x,y,z}, orientation: {x,y,z,w}}
        frame: 参考坐标系
    """
    ...

@mcp.tool()
def motion_stop(emergency: bool = False) -> dict:
    """停止所有运动"""
    ...

# ============================================
# Perception Domain (感知域)
# ============================================
@mcp.tool()
def perception_get_camera(image_type: str = "rgb") -> dict:
    """
    获取相机图像

    Args:
        image_type: "rgb", "depth", "rgbd", "segmentation"
    Returns:
        base64编码的图像数据
    """
    ...

@mcp.tool()
def perception_get_point_cloud(region: dict = None) -> dict:
    """获取点云数据"""
    ...

@mcp.tool()
def perception_detect_objects(object_type: str = None) -> dict:
    """检测场景中的物体"""
    ...

@mcp.tool()
def perception_get_laserscan() -> dict:
    """获取激光雷达扫描数据"""
    ...

# ============================================
# Planning Domain (规划域)
# ============================================
@mcp.tool()
def planning_navigate_to(goal: dict, obstacles: list = None) -> dict:
    """
    导航到目标位置

    Args:
        goal: 目标位置 {x, y, theta}
        obstacles: 动态障碍物列表
    """
    ...

@mcp.tool()
def planning_get_path(start: dict, goal: dict) -> dict:
    """获取规划路径"""
    ...

@mcp.tool()
def planning_check_reachability(pose: dict) -> dict:
    """检查位姿是否可达"""
    ...

# ============================================
# Manipulation Domain (操作域)
# ============================================
@mcp.tool()
def manipulation_grasp(object_pose: dict, grasp_type: str = "top") -> dict:
    """执行抓取"""
    ...

@mcp.tool()
def manipulation_release() -> dict:
    """释放物体"""
    ...

@mcp.tool()
def manipulation_apply_force(force: float, direction: list, duration: float) -> dict:
    """施加力/力矩"""
    ...

# ============================================
# Skill Domain (技能域)
# ============================================
@mcp.tool()
def skill_execute(skill_name: str, parameters: dict) -> dict:
    """
    执行预定义Skill

    Args:
        skill_name: Skill名称
        parameters: Skill参数
    """
    ...

@mcp.tool()
def skill_list(robot_type: str = None) -> dict:
    """列出可用Skill"""
    ...

@mcp.tool()
def skill_install(skill_name: str, version: str = "latest") -> dict:
    """从rosclaw-hub安装Skill"""
    ...

# ============================================
# System Domain (系统域)
# ============================================
@mcp.tool()
def system_get_status() -> dict:
    """获取机器人状态"""
    ...

@mcp.tool()
def system_get_joint_states() -> dict:
    """获取关节状态"""
    ...

@mcp.tool()
def system_get_diagnostics() -> dict:
    """获取诊断信息"""
    ...

@mcp.tool()
def system_emergency_stop() -> dict:
    """紧急停止"""
    ...
```

### 3.2 Tool Schema 标准

```python
# src/rosclaw/mcp/tools/schema.py
from pydantic import BaseModel, Field
from typing import Literal

class MoveBaseInput(BaseModel):
    """motion_move_base 输入参数"""
    linear_x: float = Field(
        ...,
        ge=-1.0, le=1.0,
        description="线速度 (m/s)"
    )
    angular_z: float = Field(
        ...,
        ge=-1.0, le=1.0,
        description="角速度 (rad/s)"
    )

class MoveArmInput(BaseModel):
    """motion_move_arm 输入参数"""
    joint_positions: list[float] = Field(
        ...,
        min_items=6, max_items=7,
        description="目标关节角度 (rad)"
    )
    speed: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="运动速度比例"
    )

class ToolOutput(BaseModel):
    """标准Tool输出"""
    success: bool
    message: str
    data: dict = {}
    timestamp: float
    execution_time_ms: float
```

---

## 四、高级功能

### 4.1 多机器人管理

```python
# src/rosclaw/mcp/multi_robot.py
class MultiRobotManager:
    """多机器人管理器"""

    def __init__(self):
        self.robots: Dict[str, RobotConnection] = {}

    async def register_robot(self, robot_id: str, config: dict):
        """注册机器人"""
        self.robots[robot_id] = RobotConnection(
            id=robot_id,
            type=config['type'],
            endpoint=config['endpoint'],
            capabilities=config['capabilities']
        )

    async def route_command(self, robot_id: str, tool_name: str, params: dict):
        """路由命令到指定机器人"""
        if robot_id not in self.robots:
            raise RobotNotFoundError(f"Robot {robot_id} not found")

        robot = self.robots[robot_id]

        # 检查能力
        if tool_name not in robot.capabilities:
            raise CapabilityError(f"Robot {robot_id} does not support {tool_name}")

        # 执行命令
        return await robot.execute(tool_name, params)

    async def broadcast(self, tool_name: str, params: dict):
        """广播命令到所有机器人"""
        results = {}
        for robot_id, robot in self.robots.items():
            try:
                results[robot_id] = await robot.execute(tool_name, params)
            except Exception as e:
                results[robot_id] = {"error": str(e)}
        return results
```

### 4.2 Agent-to-Agent 通信

参考 OpenClaw 的 Agent 间通信机制：

```python
# src/rosclaw/mcp/agent_comm.py
class AgentCommunication:
    """Agent间通信系统"""

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: dict
    ):
        """
        发送Agent间消息

        消息类型:
        - task_delegation: 任务委派
        - state_update: 状态更新
        - coordination: 协作协商
        - emergency: 紧急通知
        """
        message = {
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "payload": payload,
            "timestamp": time.time()
        }

        # 通过DDS广播
        await self.dds_publisher.publish(
            topic=f"/rosclaw/agents/{to_agent}",
            message=message
        )

    async def coordinate_task(
        self,
        task: str,
        agents: list[str]
    ) -> CoordinationResult:
        """
        多Agent任务协调

        示例: G1拿杯子 + UR5接杯子
        """
        # 任务分解
        subtasks = self._decompose_task(task, agents)

        # 协商分配
        assignments = await self._negotiate(subtasks, agents)

        # 建立协作会话
        session = await self._create_coordination_session(assignments)

        return CoordinationResult(session_id=session.id, assignments=assignments)
```

### 4.3 会话状态管理

```python
# src/rosclaw/mcp/session.py
class MCPSession:
    """MCP会话管理"""

    def __init__(self, session_id: str, agent_info: dict):
        self.id = session_id
        self.agent = agent_info
        self.context = SessionContext()
        self.history = []

    async def handle_tool_call(self, tool_name: str, params: dict) -> dict:
        """处理Tool调用"""
        # 记录历史
        self.history.append({
            "tool": tool_name,
            "params": params,
            "timestamp": time.time()
        })

        # 添加上下文
        enriched_params = self._enrich_with_context(params)

        # 执行Tool
        result = await self._execute_tool(tool_name, enriched_params)

        # 更新上下文
        self.context.update(tool_name, result)

        return result

    def _enrich_with_context(self, params: dict) -> dict:
        """使用会话上下文丰富参数"""
        # 添加当前机器人状态
        params['_robot_state'] = self.context.robot_state

        # 添加上一步结果
        if self.history:
            params['_previous_result'] = self.history[-1]

        return params
```

---

## 五、配置示例

### 5.1 Claude Code 配置

```json
{
  "mcpServers": {
    "rosclaw-ur5": {
      "command": "rosclaw-hub",
      "args": [
        "--robot", "ur5e",
        "--transport", "stdio",
        "--safety", "strict"
      ],
      "env": {
        "ROBOT_IP": "192.168.1.100",
        "DIGITAL_TWIN": "enabled"
      }
    },
    "rosclaw-g1": {
      "command": "rosclaw-hub",
      "args": [
        "--robot", "unitree_g1",
        "--transport", "stdio"
      ]
    }
  }
}
```

### 5.2 OpenClaw 配置

```json
{
  "mcpServers": {
    "rosclaw-embodiment": {
      "command": "rosclaw-hub",
      "args": [
        "--auto-discover",
        "--enable-multi-robot",
        "--safety", "moderate"
      ],
      "env": {
        "ROS_MASTER_URI": "http://localhost:11311",
        "CLAWHUB_API_KEY": "${CLAWHUB_KEY}"
      }
    }
  }
}
```

### 5.3 多机器人场景

```json
{
  "mcpServers": {
    "rosclaw-fleet": {
      "command": "rosclaw-hub",
      "args": [
        "--fleet-config", "/etc/rosclaw/fleet.yaml",
        "--transport", "websocket",
        "--port", "8765"
      ]
    }
  }
}
```

---

## 六、性能指标

| 指标 | 目标 | 测试方法 |
|------|------|----------|
| Tool调用延迟 | < 100ms | 本地测试 |
| 并发连接数 | > 100 | 压力测试 |
| 消息吞吐量 | > 1000 msg/s | 负载测试 |
| 数字孪生验证 | < 10ms | MJX基准 |
| 多机器人切换 | < 50ms | 场景测试 |

---

## 七、总结

| 特性 | 实现 | 优势 |
|------|------|------|
| **多传输支持** | stdio/HTTP/WebSocket | 兼容任意Agent |
| **复合工具** | 6大功能域 | 清晰的API组织 |
| **多机器人** | 动态路由+广播 |  fleet管理 |
| **Agent通信** | DDS-based | 实时协作 |
| **会话管理** | 上下文保持 | 连续交互 |
| **安全中间件** | 分层验证 | 物理安全 |

**核心设计哲学**:
> ROSClaw MCP Hub 不是简单的工具集合，而是**物理世界的通用API网关**——
> 任何Agent，通过MCP，即可掌控任意机器人。
