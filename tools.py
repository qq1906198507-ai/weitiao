#!/usr/bin/env python
# tools.py - LoRA 项目实用工具集入口
"""
LoRA 项目实用工具集

使用示例:
  python tools.py validate-data --data data/train.jsonl
  python tools.py convert alpaca --input data/alpaca.jsonl
  python tools.py list-checkpoints
  python tools.py export-checkpoint --checkpoint checkpoint-189
  python tools.py benchmark
"""

import sys
import os

# Windows 终端编码修复
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

# 添加 tools_lib 目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 项目实用工具集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  validate-data    验证训练数据格式
  convert          转换数据格式 (Alpaca/ShareGPT -> ChatML)
  list-checkpoints 列出所有 checkpoint
  export-checkpoint 导出特定 checkpoint 为可用模型
  benchmark        基准测试推理速度

详细帮助:
  python tools.py <命令> --help
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # validate-data 命令
    validate_parser = subparsers.add_parser(
        "validate-data",
        help="验证训练数据格式",
        description="验证训练数据格式是否符合 ChatML 或 Alpaca 标准"
    )
    validate_parser.add_argument("--data", default="data/train.jsonl", help="数据文件路径")

    # convert 命令
    convert_parser = subparsers.add_parser(
        "convert",
        help="转换数据格式",
        description="将 Alpaca 或 ShareGPT 格式转换为 ChatML 格式"
    )
    convert_subparsers = convert_parser.add_subparsers(dest="convert_type", help="转换类型")

    # alpaca 子命令
    alpaca_parser = convert_subparsers.add_parser("alpaca", help="Alpaca -> ChatML")
    alpaca_parser.add_argument("--input", required=True, help="输入文件路径")
    alpaca_parser.add_argument("--output", help="输出文件路径（默认自动生成）")
    alpaca_parser.add_argument("--system-prompt", default="你是一个有帮助的助手。", help="系统提示词")

    # sharegpt 子命令
    sharegpt_parser = convert_subparsers.add_parser("sharegpt", help="ShareGPT -> ChatML")
    sharegpt_parser.add_argument("--input", required=True, help="输入文件路径")
    sharegpt_parser.add_argument("--output", help="输出文件路径（默认自动生成）")

    # list-checkpoints 命令
    list_parser = subparsers.add_parser(
        "list-checkpoints",
        help="列出所有 checkpoint",
        description="列出训练输出目录中的所有 checkpoint"
    )
    list_parser.add_argument("--dir", default="output/qwen2.5-7b-lora", help="输出目录")

    # export-checkpoint 命令
    export_parser = subparsers.add_parser(
        "export-checkpoint",
        help="导出特定 checkpoint",
        description="将特定 checkpoint 导出为合并后的可用模型"
    )
    export_parser.add_argument("--checkpoint", required=True, help="checkpoint 路径或名称")
    export_parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    export_parser.add_argument("--output", help="输出目录（默认自动生成）")
    export_parser.add_argument("--quantized", action="store_true", help="使用量化加载（省显存）")

    # benchmark 命令
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="基准测试推理速度",
        description="测试模型推理速度并生成报告"
    )
    bench_parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    bench_parser.add_argument("--adapter", default="output/qwen2.5-7b-lora", help="LoRA adapter 路径")
    bench_parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # 根据命令调用对应的模块
    if args.command == "validate-data":
        from tools_lib import validate_data
        success = validate_data.validate_data(args.data)
        return 0 if success else 1

    elif args.command == "convert":
        if args.convert_type is None:
            convert_parser.print_help()
            return 1

        from tools_lib import convert_data
        if args.convert_type == "alpaca":
            return 0 if convert_data.convert_alpaca_to_chatml(
                input_path=args.input,
                output_path=args.output,
                system_prompt=args.system_prompt,
            ) else 1
        elif args.convert_type == "sharegpt":
            return 0 if convert_data.convert_sharegpt_to_chatml(
                input_path=args.input,
                output_path=args.output,
            ) else 1

    elif args.command == "list-checkpoints":
        from tools_lib import list_checkpoints
        list_checkpoints.list_checkpoints(args.dir)
        return 0

    elif args.command == "export-checkpoint":
        from tools_lib import export_checkpoint
        export_checkpoint.export_checkpoint(
            checkpoint_path=args.checkpoint,
            base_model=args.base,
            output_path=args.output,
            use_quantization=args.quantized,
        )
        return 0

    elif args.command == "benchmark":
        from tools_lib import benchmark
        benchmark.benchmark_inference(
            adapter_path=args.adapter,
            base_model=args.base,
            max_new_tokens=args.max_tokens,
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
