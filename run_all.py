"""
Transformer 课程 — 一键运行所有任务
=======================================
运行方式：
    python run_all.py              # 运行全部
    python run_all.py --step 1     # 只运行某一步
    python run_all.py --task cls   # 只运行某个任务

课程结构：
    01_basics/       基础模块逐步实现
    02_full_transformer/  完整 Transformer
    03_variants/     三种形态
    04_tasks/        三种实际任务
    utils/           工具
"""

import argparse
import os
import sys

# 确保包能被找到
sys.path.insert(0, os.path.dirname(__file__))


def run_basics():
    """运行基础模块验证（Step 1-6）"""
    print("\n" + "=" * 65)
    print("📚 01_BASICS: Transformer 基础构建块")
    print("=" * 65)

    steps = [
        ("01_embedding.py",           "Step 1: Token Embedding + Positional Encoding"),
        ("02_attention.py",           "Step 2: Scaled Dot-Product Attention"),
        ("03_multi_head_attention.py", "Step 3: Multi-Head Attention"),
        ("04_feed_forward.py",        "Step 4: Feed-Forward Network"),
        ("05_layer_norm.py",          "Step 5: Layer Norm + Residual Connection"),
        ("06_encoder_layer.py",       "Step 6: Complete Encoder Layer"),
    ]

    basics_dir = os.path.join(os.path.dirname(__file__), "01_basics")
    for filename, description in steps:
        filepath = os.path.join(basics_dir, filename)
        print(f"\n▶ {description}")
        # 动态执行文件的 __main__ 部分
        exec_file(filepath)


def run_full_transformer():
    """运行完整 Transformer 验证"""
    print("\n" + "=" * 65)
    print("🔧 02_FULL_TRANSFORMER: 完整 Transformer")
    print("=" * 65)

    transformer_dir = os.path.join(os.path.dirname(__file__), "02_full_transformer")
    sys.path.insert(0, transformer_dir)

    for filename, desc in [
        ("masks.py", "Masks 工具"),
        ("encoder.py", "Transformer Encoder"),
        ("decoder.py", "Transformer Decoder"),
        ("transformer.py", "完整 Transformer (Enc-Dec)"),
    ]:
        print(f"\n▶ {desc}")
        exec_file(os.path.join(transformer_dir, filename))


def run_variants():
    """运行三种变体验证"""
    print("\n" + "=" * 65)
    print("🔀 03_VARIANTS: 三种变体")
    print("=" * 65)

    variants_dir = os.path.join(os.path.dirname(__file__), "03_variants")
    sys.path.insert(0, variants_dir)

    for filename, desc in [
        ("only_encoder.py",    "变体1: Only Encoder (BERT 风格)"),
        ("only_decoder.py",    "变体2: Only Decoder (GPT 风格)"),
        ("encoder_decoder.py", "变体3: Encoder-Decoder (T5/BART 风格)"),
    ]:
        print(f"\n▶ {desc}")
        exec_file(os.path.join(variants_dir, filename))


def run_tasks():
    """运行三种实际任务训练"""
    import importlib.util

    print("\n" + "=" * 65)
    print("🎯 04_TASKS: 三种实际任务")
    print("=" * 65)

    tasks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_tasks")

    task_files = [
        ("classification_task", "run_classification_task", "任务1: 文本分类（Only Encoder）"),
        ("lm_task",             "run_lm_task",             "任务2: 字符级语言模型（Only Decoder）"),
        ("translation_task",    "run_translation_task",    "任务3: 序列到序列翻译（Encoder-Decoder）"),
    ]

    for module_name, func_name, desc in task_files:
        print(f"\n▶ {desc}")
        spec = importlib.util.spec_from_file_location(
            module_name,
            os.path.join(tasks_dir, f"{module_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        if tasks_dir not in sys.path:
            sys.path.insert(0, tasks_dir)
        spec.loader.exec_module(module)
        getattr(module, func_name)()


def exec_file(filepath: str):
    """动态执行 Python 文件的 __main__ 块"""
    namespace = {"__name__": "__main__", "__file__": filepath}
    module_dir = os.path.dirname(filepath)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, filepath, "exec"), namespace)


def print_course_overview():
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Transformer 从0到1 学习课程                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📂 01_basics/          基础构建块（从底层开始）               ║
║     ├── 01_embedding.py     词嵌入 + 位置编码                 ║
║     ├── 02_attention.py     缩放点积注意力                    ║
║     ├── 03_multi_head_attention.py  多头注意力               ║
║     ├── 04_feed_forward.py  前馈网络                         ║
║     ├── 05_layer_norm.py    层归一化 + 残差连接               ║
║     └── 06_encoder_layer.py 完整 Encoder 层                  ║
║                                                              ║
║  📂 02_full_transformer/    完整 Transformer                  ║
║     ├── masks.py            Mask 工具函数                    ║
║     ├── encoder.py          完整 Encoder (N 层堆叠)           ║
║     ├── decoder.py          完整 Decoder (含交叉注意力)        ║
║     └── transformer.py      Encoder-Decoder 完整版            ║
║                                                              ║
║  📂 03_variants/            三种形态                          ║
║     ├── only_encoder.py     BERT 风格（分类/NER）             ║
║     ├── only_decoder.py     GPT 风格（文本生成）               ║
║     └── encoder_decoder.py  T5/BART 风格（翻译/摘要）          ║
║                                                              ║
║  📂 04_tasks/               三种实际任务                       ║
║     ├── classification_task.py  情感分析（Only Encoder）      ║
║     ├── lm_task.py              字符级 LM（Only Decoder）     ║
║     └── translation_task.py    数字→单词翻译（Enc-Dec）        ║
║                                                              ║
║  📂 utils/                  工具模块                          ║
║     ├── visualize.py        注意力/PE 可视化                  ║
║     └── tokenizer.py        Char/Word/BPE 分词器             ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  运行方式:                                                    ║
║    python run_all.py             # 全部运行                  ║
║    python run_all.py --step 1    # 只运行 01_basics          ║
║    python run_all.py --step 2    # 只运行 02_full_transformer║
║    python run_all.py --step 3    # 只运行 03_variants        ║
║    python run_all.py --step 4    # 只运行 04_tasks           ║
║                                                              ║
║  单独运行任一文件:                                             ║
║    python 01_basics/01_embedding.py                          ║
║    python 04_tasks/translation_task.py                       ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer 课程运行器")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="1=basics, 2=full_transformer, 3=variants, 4=tasks"
    )
    args = parser.parse_args()

    print_course_overview()

    step_map = {
        1: run_basics,
        2: run_full_transformer,
        3: run_variants,
        4: run_tasks,
    }

    if args.step:
        step_map[args.step]()
    else:
        # 运行全部（基础模块 + 变体验证，任务训练可选）
        run_basics()
        run_full_transformer()
        run_variants()
        print("\n" + "=" * 65)
        print("✅ 基础验证全部完成！")
        print("\n要运行训练任务（需要一些时间），请执行：")
        print("  python run_all.py --step 4")
        print("  python 04_tasks/classification_task.py")
        print("  python 04_tasks/lm_task.py")
        print("  python 04_tasks/translation_task.py")
        print("=" * 65)

