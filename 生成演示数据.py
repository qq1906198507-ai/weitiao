# create_data.py  —— 生成一份 1000 条的演示数据
import json, random

templates = [
    {
        "instruction": "请用一句话解释什么是{}。",
        "topics": ["机器学习", "深度学习", "梯度下降", "过拟合", "正则化",
                    "卷积神经网络", "循环神经网络", "注意力机制", "Transformer", "大语言模型"],
        "answers": [
            "机器学习是让计算机从数据中自动学习规律并做出预测的技术。",
            "深度学习是基于多层神经网络的机器学习方法，能自动提取数据的层次化特征。",
            "梯度下降是一种通过沿损失函数梯度反方向迭代更新参数来最小化损失的优化算法。",
            "过拟合是模型在训练集上表现好但在新数据上泛化能力差的现象。",
            "正则化是在损失函数中添加惩罚项以防止模型过拟合的技术。",
            "卷积神经网络是利用卷积核提取局部空间特征的神经网络，广泛用于图像任务。",
            "循环神经网络是具有循环连接的神经网络，适合处理序列数据。",
            "注意力机制让模型在处理输入时能动态关注最相关的部分。",
            "Transformer是一种完全基于注意力机制的序列建模架构，是现代大模型的基础。",
            "大语言模型是参数量巨大、在海量文本上预训练的语言模型，具备强大的文本理解与生成能力。",
        ]
    },
    {
        "instruction": "将以下内容翻译成英文：{}",
        "topics": ["今天天气不错", "我喜欢编程", "人工智能正在改变世界",
                    "学习是一生的事", "数据是新时代的石油"],
        "answers": [
            "The weather is nice today.",
            "I like programming.",
            "Artificial intelligence is changing the world.",
            "Learning is a lifelong journey.",
            "Data is the oil of the new era.",
        ]
    },
]

samples = []
for _ in range(1000):
    t = random.choice(templates)
    idx = random.randint(0, len(t["topics"]) - 1)
    samples.append({
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},
            {"role": "user", "content": t["instruction"].format(t["topics"][idx])},
            {"role": "assistant", "content": t["answers"][idx]},
        ]
    })

import os; os.makedirs("data", exist_ok=True)
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"✅ 生成 {len(samples)} 条训练数据 -> data/train.jsonl")
