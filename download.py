#!/usr/bin/env python3
"""
论文快速下载脚本

用法:
    python download.py agent      # 下载 Agent 必读论文
    python download.py llm        # 下载大模型必读论文
    python download.py all        # 下载全部
"""

import urllib.request
from pathlib import Path

# Agent 必读论文 (15篇)
AGENT_PAPERS = [
    ("ReAct_Reasoning_Acting", "2210.03629"),
    ("Reflexion_Self_Reflection", "2303.11366"),
    ("Tree_of_Thoughts", "2305.08291"),
    ("MetaGPT", "2308.00352"),
    ("AutoGen", "2308.08155"),
    ("CAMEL", "2303.04637"),
    ("AgentVerse", "2308.10848"),
    ("Toolformer", "2303.04607"),
    ("Gorilla", "2305.15334"),
    ("TaskMatrix", "2303.16354"),
    ("Voyager", "2305.16291"),
    ("Generative_Agents", "2304.03442"),
    ("Chain_of_Hindsight", "2309.01762"),
    ("AgentBench", "2308.03688"),
    ("ToolEval", "2306.05527"),
]

# 大模型必读论文 (15篇)
LLM_PAPERS = [
    ("Attention_Is_All_You_Need", "1706.03762"),
    ("GPT3", "2005.14165"),
    ("LLaMA2", "2307.09288"),
    ("Chinchilla", "2203.15556"),
    ("Scaling_Laws", "2001.08361"),
    ("Mixtral_MoE", "2401.04088"),
    ("Chain_of_Thought", "2201.11903"),
    ("InstructGPT_RLHF", "2203.02155"),
    ("LoRA", "2106.09685"),
    ("Constitutional_AI", "2212.08073"),
    ("DPO", "2305.18290"),
    ("Self_Alignment", "2305.14530"),
    ("RAG", "2005.11401"),
    ("DPR", "2004.04906"),
]


def download_papers(papers, category_name):
    """下载论文列表"""
    output_dir = Path("papers") / category_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📚 下载 {category_name} 论文 ({len(papers)} 篇)")
    print("=" * 50)

    success = 0
    for name, arxiv_id in papers:
        pdf_name = f"{name}.pdf"
        pdf_path = output_dir / pdf_name

        if pdf_path.exists():
            print(f"✓ 已存在: {pdf_name}")
            success += 1
            continue

        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            print(f"↓ {pdf_name}")
            urllib.request.urlretrieve(url, pdf_path)
            success += 1
        except Exception as e:
            print(f"✗ 失败: {pdf_name} ({e})")

    print(f"\n完成: {success}/{len(papers)} 篇")
    print(f"保存位置: {output_dir.absolute()}")


def main():
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return

    target = sys.argv[1].lower()

    if target == "agent":
        download_papers(AGENT_PAPERS, "Agent")
    elif target in ["llm", "model"]:
        download_papers(LLM_PAPERS, "LLM")
    elif target == "all":
        download_papers(AGENT_PAPERS, "Agent")
        download_papers(LLM_PAPERS, "LLM")
    else:
        print(f"未知选项: {target}")
        print("可用选项: agent, llm, all")


if __name__ == "__main__":
    main()
