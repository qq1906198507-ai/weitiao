# tools/validate_data.py - 验证训练数据格式
import json
import sys
import argparse
from pathlib import Path


def validate_data(data_path: str = "data/train.jsonl"):
    """
    验证训练数据格式是否正确

    检查项：
    - JSON 格式是否合法
    - 是否包含必要的字段
    - 对话格式是否符合 ChatML
    """
    print(f"🔍 验证数据文件: {data_path}")

    if not Path(data_path).exists():
        print(f"❌ 文件不存在: {data_path}")
        return False

    errors = []
    warnings = []
    total = 0
    valid_samples = []

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"第 {i} 行: JSON 解析错误 - {e}")
                continue

            # 检查字段
            if "messages" in data:
                # ChatML 格式
                messages = data["messages"]
                if not isinstance(messages, list):
                    errors.append(f"第 {i} 行: messages 必须是数组")
                    continue

                if len(messages) == 0:
                    errors.append(f"第 {i} 行: messages 为空")
                    continue

                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"第 {i} 行, 消息 {j}: 必须是对象")
                        continue
                    if "role" not in msg:
                        errors.append(f"第 {i} 行, 消息 {j}: 缺少 role 字段")
                    if "content" not in msg:
                        errors.append(f"第 {i} 行, 消息 {j}: 缺少 content 字段")

                    role = msg.get("role")
                    if role not in ["system", "user", "assistant"]:
                        warnings.append(f"第 {i} 行: 未知 role '{role}'")

                valid_samples.append(data)

            elif "instruction" in data and "output" in data:
                # Alpaca 格式 - 转换提示
                print(f"   ⚠️ 第 {i} 行: Alpaca 格式，建议转换为 ChatML")
                warnings.append(f"第 {i} 行: Alpaca 格式")
            else:
                errors.append(f"第 {i} 行: 未知格式，需要 'messages' 或 'instruction/output'")

    # 报告结果
    print(f"\n📊 统计:")
    print(f"   总样本数: {total}")
    print(f"   有效样本: {len(valid_samples)}")
    print(f"   错误数: {len(errors)}")
    print(f"   警告数: {len(warnings)}")

    if errors:
        print(f"\n❌ 错误列表 (前10个):")
        for e in errors[:10]:
            print(f"   - {e}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
        return False

    if warnings:
        print(f"\n⚠️ 警告列表 (前5个):")
        for w in warnings[:5]:
            print(f"   - {w}")

    print(f"\n✅ 数据验证通过！")

    # 显示一个示例
    if valid_samples:
        print(f"\n📖 样例数据 (第1条):")
        sample = valid_samples[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")

    return True


def main():
    parser = argparse.ArgumentParser(description="验证训练数据格式")
    parser.add_argument("--data", default="data/train.jsonl", help="数据文件路径")
    args = parser.parse_args()

    success = validate_data(args.data)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
