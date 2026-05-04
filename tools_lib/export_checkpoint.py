# tools/export_checkpoint.py - 导出特定 checkpoint
import os
import sys
import argparse
import json
from pathlib import Path


def export_checkpoint(
    checkpoint_path: str,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    output_path: str = None,
    use_quantization: bool = False,
):
    """
    导出特定 checkpoint 为可用模型

    用途：
    - 对比不同 checkpoint 的效果
    - 选择最佳 checkpoint 部署
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # 处理 checkpoint 路径
    cp_path = Path(checkpoint_path)
    if not cp_path.exists():
        # 尝试在 output 目录下查找
        full_path = Path("output/qwen2.5-7b-lora") / checkpoint_path
        if full_path.exists():
            cp_path = full_path
        else:
            print(f"❌ 找不到 checkpoint: {checkpoint_path}")
            sys.exit(1)

    if output_path is None:
        output_path = f"output/exported-{cp_path.name}"

    print(f"🚀 导出 checkpoint: {cp_path}")
    print(f"   基础模型: {base_model}")
    print(f"   输出目录: {output_path}\n")

    # 加载 tokenizer
    print("📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    # 加载基础模型
    print("📥 加载基础模型...")
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            local_files_only=True,
        )

    # 加载 LoRA
    print(f"📥 加载 LoRA Adapter: {cp_path.name}...")
    model = PeftModel.from_pretrained(model, str(cp_path), local_files_only=True)

    # 合并权重
    print("🔧 合并权重...")
    model = model.merge_and_unload()

    # 保存
    print(f"💾 保存合并后的模型...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 保存元数据
    meta = {
        "base_model": base_model,
        "checkpoint": cp_path.name,
        "quantized": use_quantization,
    }
    with open(os.path.join(output_path, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 计算模型大小
    total_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**3)  # GB

    print(f"\n✅ 导出完成！")
    print(f"   保存位置: {output_path}")
    print(f"   模型总大小: {total_size:.2f} GB")
    print(f"\n💡 接下来你可以:")
    print(f"   1. 直接用 transformers 加载: AutoModelForCausalLM.from_pretrained('{output_path}')")
    print(f"   2. 转换为 GGUF 格式用于 llama.cpp")


def main():
    parser = argparse.ArgumentParser(description="导出特定 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="checkpoint 路径或名称")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    parser.add_argument("--output", help="输出目录（默认自动生成）")
    parser.add_argument("--quantized", action="store_true", help="使用量化加载（省显存）")
    args = parser.parse_args()

    export_checkpoint(
        checkpoint_path=args.checkpoint,
        base_model=args.base,
        output_path=args.output,
        use_quantization=args.quantized,
    )


if __name__ == "__main__":
    main()
