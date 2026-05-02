# LoRA 微调与部署使用手册

## 目录
1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [模型训练](#3-模型训练)
4. [API 部署](#4-api-部署)
5. [前端使用](#5-前端使用)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 必要依赖
```bash
pip install torch transformers datasets peft trl bitsandbytes fastapi uvicorn
```

### 1.2 确认模型缓存
确保基础模型已下载到 HuggingFace 缓存：
```bash
# Windows 默认缓存位置：
C:\Users\<用户名>\.cache\huggingface\hub\

# 查看缓存的模型
ls "C:\Users\$env:USERNAME\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct"
```

---

## 2. 数据准备

### 2.1 数据格式
创建 `data/train.jsonl`，每行一个 JSON 对象：
```json
{"instruction": "问题", "input": "", "output": "回答"}
{"instruction": "问题", "input": "", "output": "回答"}
```

### 2.2 示例数据
```bash
# 创建示例数据集
mkdir data
```

---

## 3. 模型训练

### 3.1 配置训练脚本
打开 `train.py`（或 `202605012225`），修改配置区：
```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"      # 基础模型名
DATA_PATH  = "data/train.jsonl"              # 训练数据路径
OUTPUT_DIR = "output/qwen2.5-7b-lora"        # 输出目录
NUM_EPOCHS = 3                               # 训练轮数
USE_QLORA = True                             # True=4-bit量化（省显存）
```

### 3.2 开始训练
```powershell
& "F:/python/python.exe" train.py
```

### 3.3 训练参数说明
| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `MAX_SEQ_LEN` | 最大序列长度 | 1024 (8GB显存) / 2048 (16GB显存) |
| `BATCH_SIZE` | 每设备 batch 大小 | 1-2 |
| `GRAD_ACCUM` | 梯度累积步数 | 8-16 (有效 batch = BATCH_SIZE × GRAD_ACCUM) |
| `LR` | 学习率 | 2e-4 (LoRA) |

### 3.4 训练输出
训练完成后，会在 `output/qwen2.5-7b-lora/` 目录下生成：
- `adapter_model.safetensors` - LoRA 权重
- `adapter_config.json` - LoRA 配置
- `tokenizer_config.json` - Tokenizer 配置

---

## 4. API 部署

### 4.1 启动 API 服务
```powershell
& "F:/python/python.exe" api_server.py
```

成功启动后会显示：
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4.2 API 端点

| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/chat` | 单轮对话 |
| POST | `/chat/stream` | 流式对话 (SSE) |
| POST | `/chat/history` | 多轮对话 |
| POST | `/chat/history/stream` | 多轮流式对话 |

### 4.3 请求示例

**普通请求:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "max_new_tokens": 512, "temperature": 0.7}'
```

**流式请求:**
```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: "你好",
    max_new_tokens: 512,
    temperature: 0.7,
    top_p: 0.9
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  const chunk = decoder.decode(value);
  console.log(chunk);  // data: {"text": "..."}
}
```

---

## 5. 前端使用

### 5.1 打开聊天界面
1. 确保 API 服务已启动
2. 双击 `chat.html` 用浏览器打开，或
3. VS Code 右键 "Open with Live Server"

### 5.2 界面功能
- **左侧边栏**：调整 API地址、流式模式、max tokens、temperature 等参数
- **右侧聊天区**：
  - 蓝色气泡（右侧）：用户发送的消息
  - 灰色气泡（左侧）：AI 助手回复（流式逐字显示）

### 5.3 参数调试
- **Temperature**：控制生成随机性（低=保守，高=有创意）
- **Top P**：控制采样范围（通常0.9）
- **Max Tokens**：最大生成长度

---

## 6. 常见问题

### 6.1 训练时显存不足
```python
# 方案 1：启用 QLoRA
USE_QLORA = True

# 方案 2：减小 batch size 和序列长度
BATCH_SIZE = 1
GRAD_ACCUM = 16
MAX_SEQ_LEN = 512
```

### 6.2 Windows 编码错误
如果遇到 `UnicodeDecodeError` 或 `gbk` 编码错误，运行前设置环境变量：
```powershell
$env:PYTHONUTF8=1
```

### 6.3 连接 HuggingFace 超时
使用镜像站点：
```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

或使用本地缓存（已下载的模型）：
```python
local_files_only=True  # 在 from_pretrained 中添加此参数
```

### 6.4 跨域错误 (CORS)
API 已配置允许所有来源访问。如果仍有问题，检查：
- API 地址是否正确（默认 http://localhost:8000）
- 服务是否正常运行

### 6.5 bfloat16 错误
Windows 版 bitsandbytes 不支持 bf16。已在训练脚本中禁用 bf16：
```python
fp16=True,
bf16=False,
```

---

## 附录：目录结构

```
weitiao/
├── train.py              # 训练脚本
├── api_server.py         # API 服务
├── inference.py          # 命令行推理脚本
├── chat.html             # 前端聊天界面
├── data/
│   └── train.jsonl       # 训练数据
└── output/
    └── qwen2.5-7b-lora/    # 训练输出
        ├── adapter_model.safetensors
        └── adapter_config.json
```

---

## 快速启动流程

```powershell
# 1. 训练
& "F:/python/python.exe" train.py

# 2. 启动 API
& "F:/python/python.exe" api_server.py

# 3. 打开 chat.html 开始聊天
```
