# tools.py - LoRA 项目实用工具集

## 子命令

### 1. validate-data - 验证数据格式
检查训练数据格式是否符合 ChatML 或 Alpaca 标准

```bash
python tools.py validate-data --data data/train.jsonl
```

### 2. convert - 转换数据格式

#### Alpaca -> ChatML
```bash
python tools.py convert alpaca --input data/alpaca.jsonl
```

#### ShareGPT -> ChatML
```bash
python tools.py convert sharegpt --input data/sharegpt.jsonl
```

### 3. list-checkpoints - 列出所有 checkpoint
```bash
python tools.py list-checkpoints --dir output/qwen2.5-7b-lora
```

### 4. export-checkpoint - 导出特定 checkpoint
将 checkpoint 合并为完整模型

```bash
python tools.py export-checkpoint --checkpoint checkpoint-189
```

### 5. benchmark - 基准测试推理速度
```bash
python tools.py benchmark --max-tokens 256
```

## 安装使用

1. 验证数据
```bash
python tools.py validate-data
```

2. 列出可用的 checkpoint
```bash
python tools.py list-checkpoints
```

3. 导出特定 checkpoint
```bash
python tools.py export-checkpoint --checkpoint checkpoint-189
```

4. 基准测试推理速度
```bash
python tools.py benchmark
```

## 详细帮助

```bash
python tools.py --help
python tools.py <命令> --help
```
