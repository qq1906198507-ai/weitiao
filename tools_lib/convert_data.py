# tools/convert_data.py - 数据格式转换
import json
import argparse
from pathlib import Path


def convert_alpaca_to_chatml(
    input_path: str,
    output_path: str = None,
    system_prompt: str = "你是一个有帮助的助手。",
):
    """
    将 Alpaca 格式数据转换为 ChatML 格式

    Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    ChatML: {"messages": [{"role": "...", "content": "..."}, ...]}
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('')) + "-chatml.jsonl"

    print(f"🔄 转换格式: {input_path} -> {output_path}")

    if not Path(input_path).exists():
        print(f"❌ 文件不存在: {input_path}")
        return False

    converted = 0
    errors = []

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"第 {i} 行: JSON 错误 - {e}")
                continue

            # 构建 messages
            messages = [{"role": "system", "content": system_prompt}]

            # 组合 instruction 和 input
            instruction = data.get("instruction", "")
            input_text = data.get("input", "")

            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": data.get("output", "")})

            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            converted += 1

    print(f"✅ 转换完成: {converted} 条数据")
    if errors:
        print(f"⚠️ 跳过错误: {len(errors)} 条")
    print(f"💾 保存到: {output_path}")
    return True


def convert_sharegpt_to_chatml(
    input_path: str,
    output_path: str = None,
):
    """
    将 ShareGPT 格式转换为 ChatML 格式

    ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", ...}]}
    ChatML: {"messages": [{"role": "user", "content": "..."}, ...]}
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('')) + "-chatml.jsonl"

    print(f"🔄 转换 ShareGPT -> ChatML: {input_path} -> {output_path}")

    if not Path(input_path).exists():
        print(f"❌ 文件不存在: {input_path}")
        return False

    converted = 0
    errors = []
    role_mapping = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"第 {i} 行: JSON 错误 - {e}")
                continue

            conversations = data.get("conversations", [])
            messages = []

            for conv in conversations:
                from_role = conv.get("from", "")
                value = conv.get("value", "")

                role = role_mapping.get(from_role, from_role)
                messages.append({"role": role, "content": value})

            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            converted += 1

    print(f"✅ 转换完成: {converted} 条数据")
    if errors:
        print(f"⚠️ 跳过错误: {len(errors)} 条")
    print(f"💾 保存到: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="数据格式转换工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # alpaca 命令
    alpaca_parser = subparsers.add_parser("alpaca", help="Alpaca 格式 -> ChatML")
    alpaca_parser.add_argument("--input", required=True, help="输入文件路径")
    alpaca_parser.add_argument("--output", help="输出文件路径（默认自动生成）")
    alpaca_parser.add_argument("--system-prompt", default="你是一个有帮助的助手。", help="系统提示词")

    # sharegpt 命令
    sharegpt_parser = subparsers.add_parser("sharegpt", help="ShareGPT 格式 -> ChatML")
    sharegpt_parser.add_argument("--input", required=True, help="输入文件路径")
    sharegpt_parser.add_argument("--output", help="输出文件路径（默认自动生成）")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "alpaca":
        convert_alpaca_to_chatml(
            input_path=args.input,
            output_path=args.output,
            system_prompt=args.system_prompt,
        )
    elif args.command == "sharegpt":
        convert_sharegpt_to_chatml(
            input_path=args.input,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
