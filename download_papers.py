#!/usr/bin/env python3
"""
LLM Agent 论文批量下载脚本

用法:
    python download_papers.py              # 下载所有论文
    python download_papers.py --category   # 按分类下载
    python download_papers.py --essential  # 只下载必读论文
"""

import os
import sys
from pathlib import Path
import urllib.request
from urllib.error import URLError

# 论文分类
PAPERS = {
    "01_foundational": {
        "name": "基础架构",
        "papers": [
            # Transformer
            {
                "name": "Attention_Is_All_You_Need",
                "arxiv_id": "1706.03762",
                "year": 2017,
                "essential": True,
                "notes": "Transformer 基石"
            },
            # GPT-3
            {
                "name": "GPT-3_Language_Models",
                "arxiv_id": "2005.14165",
                "year": 2020,
                "essential": True,
                "notes": "大模型涌现能力"
            },
            # Chinchilla
            {
                "name": "Chinchilla_Scaling_Laws",
                "arxiv_id": "2203.15556",
                "year": 2022,
                "essential": True,
                "notes": "计算最优缩放"
            },
            # Scaling Laws
            {
                "name": "Scaling_Laws_LLM",
                "arxiv_id": "2001.08361",
                "year": 2020,
                "essential": False,
                "notes": "缩放定律"
            },
            # LLaMA 系列没有官方 arXiv，使用技术报告链接
            {
                "name": "LLaMA_Technical_Report",
                "url": "https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/",
                "year": 2023,
                "essential": True,
                "notes": "访问官网获取"
            },
        ]
    },
    "02_agent_core": {
        "name": "Agent 核心",
        "papers": [
            # ReAct
            {
                "name": "ReAct_Reasoning_Acting",
                "arxiv_id": "2210.03629",
                "year": 2023,
                "essential": True,
                "notes": "推理+行动范式"
            },
            # Reflexion
            {
                "name": "Reflexion_Self_Reflection",
                "arxiv_id": "2303.11366",
                "year": 2023,
                "essential": True,
                "notes": "自我反思机制"
            },
            # MetaGPT
            {
                "name": "MetaGPT",
                "arxiv_id": "2308.00352",
                "year": 2023,
                "essential": True,
                "notes": "SOP 多 Agent 框架"
            },
            # AutoGen
            {
                "name": "AutoGen",
                "arxiv_id": "2308.08155",
                "year": 2023,
                "essential": True,
                "notes": "微软多 Agent 框架"
            },
            # Voyager
            {
                "name": "Voyager_Embedded_Agent",
                "arxiv_id": "2305.16291",
                "year": 2023,
                "essential": False,
                "notes": "具身 Agent + 技能库"
            },
            # CAMEL
            {
                "name": "CAMEL_Communicative_Agents",
                "arxiv_id": "2303.04637",
                "year": 2023,
                "essential": False,
                "notes": "多 Agent 通信"
            },
            # Generative Agents
            {
                "name": "Generative_Agents",
                "arxiv_id": "2304.03442",
                "year": 2023,
                "essential": False,
                "notes": "社交模拟 Agent"
            },
            # HuggingGPT
            {
                "name": "HuggingGPT",
                "arxiv_id": "2303.17580",
                "year": 2023,
                "essential": False,
                "notes": "模型编排"
            },
        ]
    },
    "03_reasoning": {
        "name": "推理与规划",
        "papers": [
            # Chain-of-Thought
            {
                "name": "Chain_of_Thought_CoT",
                "arxiv_id": "2201.11903",
                "year": 2023,
                "essential": True,
                "notes": "思维链推理"
            },
            # Tree of Thoughts
            {
                "name": "Tree_of_Thoughts_ToT",
                "arxiv_id": "2305.08291",
                "year": 2024,
                "essential": True,
                "notes": "树状搜索推理"
            },
            # Self-Consistency
            {
                "name": "Self_Consistency",
                "arxiv_id": "2203.11171",
                "year": 2023,
                "essential": False,
                "notes": "多路径验证"
            },
            # Least-to-Most
            {
                "name": "Least_to_Most",
                "arxiv_id": "2210.14709",
                "year": 2023,
                "essential": False,
                "notes": "分步分解"
            },
            # Graph of Thoughts
            {
                "name": "Graph_of_Thoughts",
                "arxiv_id": "2308.09719",
                "year": 2023,
                "essential": False,
                "notes": "图结构推理"
            },
            # RAP
            {
                "name": "RAP_Reasoning_Planning",
                "arxiv_id": "2309.10073",
                "year": 2024,
                "essential": False,
                "notes": "规划即推理"
            },
        ]
    },
    "04_tool_use": {
        "name": "工具使用",
        "papers": [
            # Toolformer
            {
                "name": "Toolformer",
                "arxiv_id": "2303.04607",
                "year": 2023,
                "essential": True,
                "notes": "自主工具学习"
            },
            # Gorilla
            {
                "name": "Gorilla_API_Calling",
                "arxiv_id": "2305.15334",
                "year": 2023,
                "essential": True,
                "notes": "精确 API 调用"
            },
            # API-Bank
            {
                "name": "API_Bank_Benchmark",
                "arxiv_id": "2304.08244",
                "year": 2023,
                "essential": False,
                "notes": "工具调用基准"
            },
            # ToolAlpaca
            {
                "name": "ToolAlpaca",
                "arxiv_id": "2306.07937",
                "year": 2023,
                "essential": False,
                "notes": "通用工具学习"
            },
            # Chameleon
            {
                "name": "Chameleon_Plug_Play",
                "arxiv_id": "2304.09842",
                "year": 2023,
                "essential": False,
                "notes": "即插即用推理"
            },
        ]
    },
    "05_alignment": {
        "name": "对齐与安全",
        "papers": [
            # InstructGPT
            {
                "name": "InstructGPT_RLHF",
                "arxiv_id": "2203.02155",
                "year": 2022,
                "essential": True,
                "notes": "RLHF 基石"
            },
            # Constitutional AI
            {
                "name": "Constitutional_AI",
                "arxiv_id": "2212.08073",
                "year": 2022,
                "essential": True,
                "notes": "CAI / RLAIF"
            },
            # DPO
            {
                "name": "DPO_Direct_Preference",
                "arxiv_id": "2305.18290",
                "year": 2023,
                "essential": True,
                "notes": "直接偏好优化"
            },
            # RRHF
            {
                "name": "RRHF",
                "arxiv_id": "2304.05374",
                "year": 2023,
                "essential": False,
                "notes": "排序响应"
            },
            # KTO
            {
                "name": "KTO",
                "arxiv_id": "2402.01306",
                "year": 2024,
                "essential": False,
                "notes": "无参考对齐"
            },
            # Llama 2 对齐
            {
                "name": "Llama2_Alignment",
                "arxiv_id": "2307.09288",
                "year": 2023,
                "essential": False,
                "notes": "安全对齐实践"
            },
        ]
    },
    "06_rag": {
        "name": "RAG 检索增强",
        "papers": [
            # RAG 原始论文
            {
                "name": "RAG_Original",
                "arxiv_id": "2005.11401",
                "year": 2020,
                "essential": True,
                "notes": "RAG 基石"
            },
            # DPR
            {
                "name": "DPR_Dense_Passage",
                "arxiv_id": "2004.04906",
                "year": 2020,
                "essential": True,
                "notes": "密集段落检索"
            },
            # Self-RAG
            {
                "name": "Self_RAG",
                "arxiv_id": "2310.11511",
                "year": 2024,
                "essential": False,
                "notes": "自主 RAG"
            },
            # HyDE
            {
                "name": "HyDE",
                "arxiv_id": "2212.10496",
                "year": 2022,
                "essential": False,
                "notes": "假设文档嵌入"
            },
            # RAPTOR
            {
                "name": "RAPTOR",
                "arxiv_id": "2401.18059",
                "year": 2024,
                "essential": False,
                "notes": "递归摘要检索"
            },
            # GraphRAG
            {
                "name": "GraphRAG",
                "arxiv_id": "2404.16130",
                "year": 2024,
                "essential": False,
                "notes": "知识图谱增强"
            },
        ]
    },
    "07_evaluation": {
        "name": "评估与基准",
        "papers": [
            # AgentBench
            {
                "name": "AgentBench",
                "arxiv_id": "2308.03688",
                "year": 2023,
                "essential": False,
                "notes": "Agent 综合评估"
            },
            # ToolEval
            {
                "name": "ToolEval",
                "arxiv_id": "2306.05527",
                "year": 2023,
                "essential": False,
                "notes": "工具使用评估"
            },
            # MT-Bench
            {
                "name": "MT_Bench",
                "arxiv_id": "2306.05685",
                "year": 2023,
                "essential": False,
                "notes": "多轮对话评估"
            },
            # HumanEval
            {
                "name": "HumanEval",
                "arxiv_id": "2107.03374",
                "year": 2021,
                "essential": False,
                "notes": "代码生成基准"
            },
            # GSM8K
            {
                "name": "GSM8K",
                "arxiv_id": "2110.14168",
                "year": 2022,
                "essential": False,
                "notes": "数学推理数据集"
            },
            # MMLU
            {
                "name": "MMLU",
                "arxiv_id": "2009.03300",
                "year": 2020,
                "essential": False,
                "notes": "多任务理解"
            },
        ]
    },
    "08_latest_2024_2025": {
        "name": "最新进展",
        "papers": [
            # DeepSeek-R1
            {
                "name": "DeepSeek_R1",
                "arxiv_id": "2501.19393",
                "year": 2025,
                "essential": True,
                "notes": "纯RL训练推理"
            },
            # OpenAI o1
            {
                "name": "OpenAI_o1_Series",
                "url": "https://openai.com/index/learning-to-reason-with-llms/",
                "year": 2024,
                "essential": True,
                "notes": "推理模型, 官网"
            },
            # Llama 3.1
            {
                "name": "Llama_3_1",
                "arxiv_id": "2407.21783",
                "year": 2024,
                "essential": False,
                "notes": "开源最强"
            },
            # Qwen2
            {
                "name": "Qwen2",
                "arxiv_id": "2307.08588",
                "year": 2024,
                "essential": False,
                "notes": "强大多语言"
            },
            # SWE-agent
            {
                "name": "SWE_agent",
                "arxiv_id": "2405.15793",
                "year": 2024,
                "essential": False,
                "notes": "软件工程 Agent"
            },
        ]
    }
}


def download_paper(paper, output_dir):
    """下载单篇论文"""
    pdf_name = f"{paper['name']}.pdf"
    pdf_path = output_dir / pdf_name

    # 跳过已下载
    if pdf_path.exists():
        print(f"  ✓ 已存在: {pdf_name}")
        return True

    # 构建 PDF URL
    if paper.get("url"):
        url = paper["url"]
        if not url.endswith(".pdf"):
            print(f"  ⚠️  需手动访问: {paper['name']} -> {url}")
            return False
    else:
        arxiv_id = paper.get("arxiv_id")
        if arxiv_id:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        else:
            print(f"  ✗ 无有效链接: {paper['name']}")
            return False

    # 下载
    try:
        print(f"  ↓ 下载中: {pdf_name}")
        urllib.request.urlretrieve(url, pdf_path)
        print(f"  ✓ 完成: {pdf_name}")
        return True
    except URLError as e:
        print(f"  ✗ 下载失败: {pdf_name} - {e}")
        return False
    except Exception as e:
        print(f"  ✗ 错误: {pdf_name} - {e}")
        return False


def list_categories():
    """列出所有分类"""
    print("\n📚 论文分类:")
    print("=" * 50)
    for cat_id, cat_info in PAPERS.items():
        essential_count = sum(1 for p in cat_info["papers"] if p.get("essential", False))
        total_count = len(cat_info["papers"])
        print(f"\n[{cat_id}] {cat_info['name']}")
        print(f"    必读: {essential_count} 篇 | 总计: {total_count} 篇")
        for paper in cat_info["papers"]:
            essential = "🔥" if paper.get("essential") else "⭐"
            print(f"    {essential} {paper['name']} ({paper['year']}) - {paper['notes']}")


def download_all(essential_only=False):
    """下载所有论文"""
    base_dir = Path("papers")

    if essential_only:
        print("\n🔥 下载必读论文...")
    else:
        print("\n📚 下载所有论文...")

    total = 0
    success = 0
    manual = 0

    for cat_id, cat_info in PAPERS.items():
        category_dir = base_dir / cat_info["name"]
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{cat_info['name']}]")

        for paper in cat_info["papers"]:
            if essential_only and not paper.get("essential"):
                continue

            total += 1
            result = download_paper(paper, category_dir)
            if result:
                success += 1
            else:
                # 检查是否是手动访问的 URL
                if paper.get("url") and not paper["url"].endswith(".pdf"):
                    manual += 1

    print("\n" + "=" * 50)
    print(f"✓ 下载完成: {success}/{total} 篇")
    if manual > 0:
        print(f"⚠️  有 {manual} 篇需要手动访问官网获取")
    print(f"📁 保存位置: {base_dir.absolute()}")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-c", "--category", "list"]:
            list_categories()
        elif sys.argv[1] in ["-e", "--essential"]:
            download_all(essential_only=True)
        elif sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
        else:
            print("未知选项，使用 --help 查看帮助")
    else:
        # 默认: 先列出分类，然后下载全部
        list_categories()
        print("\n" + "=" * 50)
        response = input("\n是否开始下载所有论文? (y/n, 或输入 'e' 只下载必读): ").strip().lower()

        if response == "e":
            download_all(essential_only=True)
        elif response == "y":
            download_all(essential_only=False)
        else:
            print("取消下载")


if __name__ == "__main__":
    main()
