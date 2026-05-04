# tools/list_checkpoints.py - 列出所有 checkpoint
import os
import json
import argparse
from pathlib import Path


def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def list_checkpoints(output_dir: str = "output/qwen2.5-7b-lora"):
    """
    列出所有可用的 checkpoint
    """
    print(f"📂 检查目录: {output_dir}\n")

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"❌ 目录不存在: {output_dir}")
        return

    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            checkpoints.append(item)

    # 按步数排序
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))

    # 检查是否有最终模型
    final_model_exists = (output_path / "adapter_model.safetensors").exists()

    if not checkpoints and not final_model_exists:
        print("⚠️ 没有找到 checkpoint 或最终模型")
        return

    print("📋 发现的模型:")
    print("-" * 60)

    # 显示最终模型
    if final_model_exists:
        config_file = output_path / "adapter_config.json"
        total_size = sum(
            f.stat().st_size for f in output_path.iterdir() if f.is_file()
        )
        print(f"\n   🏆 最终模型 (final)")
        print(f"      目录: {output_dir}")
        print(f"      大小: {format_size(total_size)}")

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"      LoRA r: {config.get('r', 'N/A')}")
                print(f"      LoRA alpha: {config.get('lora_alpha', 'N/A')}")
                print(f"      目标模块: {', '.join(config.get('target_modules', [])[:3])}...")
            except Exception:
                pass

    # 显示 checkpoints
    if checkpoints:
        print(f"\n   📁 Checkpoints ({len(checkpoints)} 个):")
        for cp in checkpoints:
            # 获取文件大小
            total_size = sum(
                f.stat().st_size for f in cp.rglob("*") if f.is_file()
            )

            # 尝试读取训练状态
            trainer_state = cp / "trainer_state.json"
            epoch_info = ""
            if trainer_state.exists():
                try:
                    with open(trainer_state, 'r') as f:
                        state = json.load(f)
                    epoch = state.get('epoch', 0)
                    global_step = state.get('global_step', 0)
                    epoch_info = f" | epoch={epoch:.2f}, step={global_step}"
                except Exception:
                    pass

            print(f"      {cp.name:20} {format_size(total_size):>10}{epoch_info}")

    # 显示使用示例
    if checkpoints:
        print(f"\n💡 使用示例:")
        latest = checkpoints[-1].name
        print(f"   # 导出特定 checkpoint")
        print(f"   python tools/export_checkpoint.py --checkpoint {latest}")
        print(f"\n   # 使用 checkpoint 进行推理")
        print(f"   python inference.py  # 修改 ADAPTER_PATH = 'output/qwen2.5-7b-lora/{latest}'")


def main():
    parser = argparse.ArgumentParser(description="列出所有 checkpoint")
    parser.add_argument("--dir", default="output/qwen2.5-7b-lora", help="输出目录")
    args = parser.parse_args()

    list_checkpoints(args.dir)


if __name__ == "__main__":
    main()
