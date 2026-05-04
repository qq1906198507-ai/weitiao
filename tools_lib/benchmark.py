# tools/benchmark.py - 基准测试推理速度
import time
import json
import argparse
from pathlib import Path


def benchmark_inference(
    adapter_path: str = "output/qwen2.5-7b-lora",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    test_prompts: list = None,
    max_new_tokens: int = 256,
):
    """
    基准测试推理速度
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if test_prompts is None:
        test_prompts = [
            "你好",
            "请解释什么是机器学习",
            "写一个快速排序算法",
            "翻译：Artificial Intelligence is transforming the world.",
        ]

    print(f"⏱️ 开始推理基准测试...")
    print(f"   模型: {adapter_path}")
    print(f"   max_new_tokens: {max_new_tokens}\n")

    # 加载模型
    print("📥 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("📥 加载基础模型...")
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
        local_files_only=True,
    )

    print(f"📥 加载 LoRA Adapter...")

    # 处理路径
    adapter = Path(adapter_path)
    if not adapter.exists():
        adapter = Path("output/qwen2.5-7b-lora") / adapter_path

    model = PeftModel.from_pretrained(model, str(adapter), local_files_only=True)
    model.eval()

    device = next(model.parameters()).device
    print(f"\n🔋 设备: {device}")
    print(f"📊 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("")

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        # 预热 - 一次生成
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)

        # 正式测试
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - start

        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0

        results.append({
            "prompt": prompt[:40] + "..." if len(prompt) > 40 else prompt,
            "tokens": int(num_tokens),
            "time": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
        })

        print(f"   [{i}/{len(test_prompts)}] {prompt[:30]}...")
        print(f"       生成 {num_tokens} tokens | 耗时 {elapsed:.2f}s | 速度 {tokens_per_sec:.2f} t/s")

    # 汇总
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_tokens = sum(r["tokens"] for r in results) / len(results)
    avg_speed = sum(r["tokens_per_sec"] for r in results) / len(results)

    print(f"\n" + "="*50)
    print(f"📊 测试汇总")
    print(f"   平均耗时: {avg_time:.2f}s")
    print(f"   平均生成: {avg_tokens:.0f} tokens")
    print(f"   平均速度: {avg_speed:.2f} tokens/sec")
    print(f"   推论: ", end="")

    if avg_speed > 50:
        print("⚡ 非常快（高端 GPU）")
    elif avg_speed > 20:
        print("💪 良好（中端 GPU）")
    elif avg_speed > 10:
        print("✅ 可接受（入门 GPU）")
    else:
        print("😢 较慢（可能需要 CPU 或等待更久）")

    # 保存结果
    result_file = Path("output/benchmark_results.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump({
            "model": str(adapter_path),
            "base_model": base_model,
            "max_new_tokens": max_new_tokens,
            "results": results,
            "summary": {
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "avg_tokens_per_sec": avg_speed,
            }
        }, f, indent=2)

    print(f"\n💾 结果保存到: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="基准测试推理速度")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct", help="基础模型路径")
    parser.add_argument("--adapter", default="output/qwen2.5-7b-lora", help="LoRA adapter 路径")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    args = parser.parse_args()

    benchmark_inference(
        adapter_path=args.adapter,
        base_model=args.base,
        max_new_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
