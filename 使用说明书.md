# LoRA 微调与部署完整使用手册

## 目录
1. [项目概述](#1-项目概述)
2. [环境准备](#2-环境准备)
3. [快速开始](#3-快速开始)
4. [数据准备](#4-数据准备)
5. [模型训练](#5-模型训练)
6. [模型推理](#6-模型推理)
7. [API 部署](#7-api-部署)
8. [Web 界面使用](#8-web-界面使用)
9. [工具集使用](#9-工具集使用)
10. [常见问题](#10-常见问题)

---

## 1. 项目概述

本项目是一个完整的 **Qwen2.5-7B LoRA 微调解决方案**，包含：

| 模块 | 功能 | 文件 |
|------|------|------|
| 数据生成 | 生成演示训练数据 | `生成演示数据.py` |
| 模型训练 | LoRA/QLoRA 微调 | `训练模型.py` |
| 终端推理 | 命令行交互 | `inference.py` |
| API 服务 | FastAPI 后端 | `api_server.py` |
| Web 界面 | 浏览器聊天 | `chat.html` |
| 工具集 | 数据验证、导出、测试 | `tools.py` |

**技术栈：**
- 基础模型：Qwen/Qwen2.5-7B-Instruct (70亿参数)
- 微调方法：LoRA (低秩适配)
- 量化方案：QLoRA 4-bit (显存占用仅 ~6GB)
- 后端框架：FastAPI + Uvicorn
- 前端：纯 HTML/CSS/JS

---

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install torch transformers datasets peft trl bitsandbytes fastapi uvicorn
```

### 2.2 确认模型缓存

确保基础模型已下载到本地：

```bash
# Windows 默认缓存位置
C:\Users\<用户名>\.cache\huggingface\hub\

# 查看缓存的模型
ls "C:\Users\$env:USERNAME\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct"
```

如果未下载，可以先运行推理脚本自动下载，或设置镜像：
```powershell
# PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 2.3 显存要求

| 配置 | 显存需求 | 适用场景 |
|------|---------|---------|
| QLoRA (4-bit) | 6-8 GB | RTX 3060/4060 |
| LoRA (8-bit) | 12-14 GB | RTX 3080/4070 |
| LoRA (16-bit) | 16+ GB | RTX 4090 / A100 |

---

## 3. 快速开始

### 3.1 完整流程（3 步走）

```bash
# 第 1 步：生成演示数据
python 生成演示数据.py

# 第 2 步：训练模型（约 30 分钟，视 GPU 而定）
python 训练模型.py

# 第 3 步：启动 API 服务
python api_server.py
```

然后双击打开 `chat.html` 即可开始聊天。

---

## 4. 数据准备

### 4.1 数据格式

本项目使用 **ChatML 格式**（OpenAI 兼容）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是让计算机从数据中自动学习规律的技术。"}
  ]
}
```

保存为 `data/train.jsonl`（每行一条 JSON）。

### 4.2 从 Alpaca 格式转换

如果你的数据是 Alpaca 格式：
```json
{"instruction": "解释机器学习", "input": "", "output": "机器学习是..."}
```

使用工具转换：
```bash
python tools.py convert alpaca --input data/alpaca.jsonl --output data/train.jsonl
```

### 4.3 验证数据格式

```bash
python tools.py validate-data --data data/train.jsonl
```

输出示例：
```
🔍 验证数据文件: data/train.jsonl

📊 统计:
   总样本数: 1000
   有效样本: 1000
   错误数: 0
   警告数: 0

✅ 数据验证通过！
```

---

## 5. 模型训练

### 5.1 配置参数

编辑 `训练模型.py` 的配置区：

```python
# ============================================================
# 配置区 —— 你只需要改这里
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型路径
DATA_PATH  = "data/train.jsonl"          # 训练数据路径
OUTPUT_DIR = "output/qwen2.5-7b-lora"    # 输出目录
MAX_SEQ_LEN = 1024        # 序列长度
NUM_EPOCHS  = 3           # 训练轮数
BATCH_SIZE  = 2           # 每卡 batch
GRAD_ACCUM  = 8           # 梯度累积步数
LR = 2e-4                 # 学习率
USE_QLORA = True          # True=省显存，False=更快
# ============================================================
```

### 5.2 启动训练

```bash
python 训练模型.py
```

训练过程输出：
```
🚀 开始训练...
📁 输出目录: output/qwen2.5-7b-lora
📊 有效 batch size: 16

trainable params: 20,971,520 || all params: 7,636,889,600 || trainable%: 0.2743%

[10/100 00:15<02:30, 0.34 it/s, epoch 0.3/3, loss 1.234]
```

### 5.3 训练输出

训练完成后，输出目录结构：

```
output/qwen2.5-7b-lora/
├── adapter_config.json           # LoRA 配置
├── adapter_model.safetensors    # LoRA 权重 (~20MB)
├── tokenizer.json               # Tokenizer
├── checkpoint-126/              # 中间 checkpoint
│   ├── adapter_model.safetensors
│   └── trainer_state.json
└── checkpoint-189/              # 最终 checkpoint
    └── ...
```

### 5.4 显存不足解决方案

如果遇到 `OutOfMemoryError`：

```python
# 方案 1：启用 QLoRA
USE_QLORA = True

# 方案 2：减小序列长度
MAX_SEQ_LEN = 512

# 方案 3：减小 batch size
BATCH_SIZE = 1
GRAD_ACCUM = 16  # 保持有效 batch = 16

# 方案 4：梯度检查点（已默认启用）
gradient_checkpointing=True
```

---

## 6. 模型推理

### 6.1 终端交互式推理

```bash
python inference.py
```

交互示例：
```
正在加载模型...
✅ 模型加载完成，开始推理（输入 'exit' 退出）

你: 什么是过拟合
助手: 过拟合是模型在训练集上表现好但在新数据上泛化能力差的现象。

你: exit
再见！
```

### 6.2 推理参数调优

编辑 `inference.py` 调整生成参数：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,    # 最大生成长度
    do_sample=True,        # 采样模式
    temperature=0.7,       # 温度（创意性）
    top_p=0.9,             # 核采样
    repetition_penalty=1.1, # 重复惩罚
)
```

| 参数 | 低值 | 高值 |
|------|------|------|
| temperature | 保守、确定性强 | 创意、随机性强 |
| top_p | 词汇范围窄 | 词汇范围广 |
| repetition_penalty | 容易重复 | 避免重复 |

---

## 7. API 部署

### 7.1 启动 API 服务

```bash
python api_server.py
```

启动成功输出：
```
🚀 正在加载模型...
✅ 模型加载完成，准备就绪！
📁 Adapter: output/qwen2.5-7b-lora
🔌 设备: cuda:0

🚀 启动 API 服务: http://localhost:8000
   API 文档: http://localhost:8000/docs
```

### 7.2 API 端点列表

| 方法 | 端点 | 功能 | 说明 |
|------|------|------|------|
| GET | `/` | 健康检查 | 返回服务状态 |
| POST | `/chat` | 单轮对话 | 非流式响应 |
| POST | `/chat/stream` | 流式对话 | SSE 逐字输出 |
| POST | `/chat/history` | 多轮对话 | 支持上下文 |
| POST | `/chat/history/stream` | 多轮流式 | 流式+上下文 |

### 7.3 API 调用示例

**普通请求：**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "什么是深度学习？",
    "max_new_tokens": 512,
    "temperature": 0.7
  }'
```

响应：
```json
{
  "response": "深度学习是一种基于多层神经网络的机器学习方法..."
}
```

**流式请求：**
```javascript
const response = await fetch('http://localhost:8000/chat/stream', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: "你好",
    max_new_tokens: 512,
    temperature: 0.7
  })
});

const reader = response.body.getReader();
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  // 逐字显示...
}
```

**多轮对话：**
```bash
curl -X POST http://localhost:8000/chat/history \
  -H "Content-Type: application/json" \
  -d '[
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮助你？"},
    {"role": "user", "content": "什么是机器学习？"}
  ]'
```

---

## 8. Web 界面使用

### 8.1 打开界面

1. 确保 API 服务已启动 (`python api_server.py`)
2. 双击 `chat.html` 用浏览器打开，或
3. 在 VS Code 中右键 → "Open with Live Server"

### 8.2 界面功能

```
┌─────────────────────────────────────────────────────────┐
│ 🤖 LoRA Fine-tuned Chat          ● 在线 (Qwen2.5-7B)    │  ← 头部状态栏
├────────────────┬──────────────────────────────────────┤
│ 📝 新对话       │                                      │
│                │  🤖 你好！有什么可以帮助你？           │  ← AI 消息（左）
│ API 地址       │                                      │
│ localhost:8000 │  👤 什么是机器学习？                  │  ← 用户消息（右）
│                │                                      │
│ ☑️ 流式输出    │  🤖 机器学习是让计算机...             │
│                │                                      │
│ Max Tokens: 512│                                      │
│ [────────────] │  ┌─────────────────────────────┐      │
│                │  │ 输入消息... (Enter 发送)    │ [发送] │  ← 输入区
│ Temperature: 0.7                                    │
│ [────────────] │                                      │
│                │                                      │
│ Top P: 0.9     │                                      │
│ [────────────] │                                      │
│                │                                      │
│ 🔄 测试连接    │                                      │
└────────────────┴──────────────────────────────────────┘
        ↑ 左侧设置面板              ↑ 聊天区域
```

### 8.3 参数调节说明

| 参数 | 滑块范围 | 默认值 | 效果 |
|------|---------|--------|------|
| Max Tokens | 64-2048 | 512 | 控制回答长度 |
| Temperature | 0-2 | 0.7 | 创意性，越高越随机 |
| Top P | 0-1 | 0.9 | 词汇多样性 |
| 流式输出 | 开关 | 开 | 逐字显示效果 |

---

## 9. 工具集使用

### 9.1 工具列表

```bash
python tools.py --help
```

| 命令 | 功能 | 需要 GPU |
|------|------|---------|
| `validate-data` | 验证数据格式 | ❌ |
| `convert` | 格式转换 | ❌ |
| `list-checkpoints` | 列出 checkpoints | ❌ |
| `export-checkpoint` | 导出合并模型 | ✅ |
| `benchmark` | 基准测试 | ✅ |

### 9.2 验证数据

```bash
python tools.py validate-data --data data/train.jsonl
```

### 9.3 列出 Checkpoints

```bash
python tools.py list-checkpoints
```

输出示例：
```
📋 发现的模型:
------------------------------------------------------------

   🏆 最终模型 (final)
      目录: output/qwen2.5-7b-lora
      大小: 125.3 MB
      LoRA r: 16
      LoRA alpha: 32

   📁 Checkpoints (2 个):
      checkpoint-126        125.3 MB | epoch=1.50, step=126
      checkpoint-189        125.3 MB | epoch=2.25, step=189
```

### 9.4 导出 Checkpoint

将特定 checkpoint 合并为基础模型格式：

```bash
# 导出 checkpoint-189
python tools.py export-checkpoint --checkpoint checkpoint-189

# 导出到指定目录
python tools.py export-checkpoint --checkpoint checkpoint-189 --output my-model
```

用途：
- 部署到不支持 PEFT 的平台
- 转换为 GGUF 格式用于 llama.cpp
- 提高推理速度（无需动态合并）

### 9.5 基准测试

测试模型推理速度：

```bash
python tools.py benchmark --max-tokens 256
```

输出示例：
```
⏱️ 开始推理基准测试...

[1/4] 你好...
    生成 42 tokens | 耗时 1.23s | 速度 34.15 t/s

📊 测试汇总
   平均耗时: 1.45s
   平均生成: 128 tokens
   平均速度: 28.50 tokens/sec
   推论: ⚡ 非常快（高端 GPU）
```

---

## 10. 常见问题

### Q1: Windows 编码错误
**症状：** `UnicodeDecodeError: 'gbk' codec can't decode`

**解决：**
```powershell
$env:PYTHONUTF8=1
```

### Q2: 连接 HuggingFace 超时
**症状：** `Connection timeout` 或 `SSL error`

**解决：**
```powershell
# 使用镜像站
$env:HF_ENDPOINT="https://hf-mirror.com"

# 或仅使用本地缓存
# 在代码中添加: local_files_only=True
```

### Q3: 显存不足 (OOM)
**症状：** `torch.cuda.OutOfMemoryError`

**解决：**
1. 启用 QLoRA: `USE_QLORA = True`
2. 减小 `MAX_SEQ_LEN` 到 512
3. 减小 `BATCH_SIZE` 到 1
4. 增加 `GRAD_ACCUM` 保持有效 batch

### Q4: API 启动失败
**症状：** `Port 8000 is already in use`

**解决：**
```python
# 修改 api_server.py 中的端口
PORT = 8001  # 换一个端口
```

### Q5: 前端无法连接 API
**症状：** 浏览器显示 "离线"

**排查：**
1. 确认 API 服务已启动
2. 检查 API 地址是否正确（默认 `http://localhost:8000`）
3. 点击 "🔄 测试连接" 按钮
4. 检查浏览器控制台是否有 CORS 错误

### Q6: 生成结果质量差
**解决方案：**
1. 增加训练数据量（建议 >1000 条）
2. 增加训练轮数 `NUM_EPOCHS = 5`
3. 调整生成参数：降低 `temperature` 到 0.5
4. 尝试不同的 checkpoint

---

## 附录：目录结构

```
weitiao/
├── README.md                    # 本手册
├── 训练模型.py                 # 训练脚本
├── 生成演示数据.py             # 数据生成
├── inference.py                # 终端推理
├── api_server.py               # API 服务
├── chat.html                   # Web 界面
├── tools.py                    # 工具集入口
├── tools_lib/                  # 工具模块
│   ├── validate_data.py
│   ├── convert_data.py
│   ├── list_checkpoints.py
│   ├── export_checkpoint.py
│   └── benchmark.py
├── data/
│   └── train.jsonl             # 训练数据
└── output/
    └── qwen2.5-7b-lora/        # 训练输出
        ├── adapter_model.safetensors
        ├── checkpoint-126/
        └── checkpoint-189/
```

---

## 相关链接

- [Transformers 文档](https://huggingface.co/docs/transformers)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Qwen 模型](https://huggingface.co/Qwen)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
