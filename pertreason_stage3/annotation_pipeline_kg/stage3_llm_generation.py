"""
Stage 3: LLM 文本生成
使用 vLLM API 将路径转换为自然语言推理
"""

import json
import logging
from typing import Dict, List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class Stage3LLMGeneration:
    """Stage 3: LLM 文本生成"""

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        min_reasoning_length: int = 20,
        max_reasoning_length: int = 200,
        blacklist_keywords: Optional[List[str]] = None
    ):
        """
        Args:
            llm_base_url: vLLM server URL
            llm_model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            min_reasoning_length: 最小推理长度（单词数）
            max_reasoning_length: 最大推理长度（单词数）
            blacklist_keywords: 黑名单关键词
        """
        self.client = OpenAI(
            base_url=llm_base_url,
            api_key="EMPTY"  # vLLM 不需要真实 API key
        )
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_reasoning_length = min_reasoning_length
        self.max_reasoning_length = max_reasoning_length
        self.blacklist_keywords = blacklist_keywords or []

    def build_prompt(
        self,
        perturbation: str,
        target_gene: str,
        label: str,
        triplets: List[tuple],
        basal_context: Dict[str, float]
    ) -> str:
        """
        构建 LLM prompt

        Args:
            perturbation: 扰动名称
            target_gene: 目标基因
            label: 标签
            triplets: 三元组路径
            basal_context: 基础表达上下文

        Returns:
            Prompt 字符串
        """
        # 格式化路径
        path_str = " → ".join([
            f"{source} --[{relation}]--> {target}"
            for source, relation, target in triplets
        ])

        # 提取路径中涉及的基因表达
        genes_in_path = set()
        for source, _, target in triplets:
            genes_in_path.add(source)
            genes_in_path.add(target)

        basal_str = ", ".join([
            f"{gene}: {basal_context.get(gene, 0.0):.3f}"
            for gene in sorted(genes_in_path)
            if gene in basal_context
        ])

        # 构建 prompt
        prompt = f"""You are a systems biology expert. Given a perturbation and a biological pathway, generate a concise mechanistic reasoning explaining how the perturbation affects the target gene.

**Perturbation**: {perturbation}
**Target Gene**: {target_gene}
**Observed Effect**: {label}
**Pathway**: {path_str}
**Basal Expression**: {basal_str}

Generate a clear, scientifically accurate reasoning (50-150 words) explaining the mechanistic relationship. Focus on the biological mechanisms and avoid uncertain language.

**Reasoning**:"""

        return prompt

    def generate_reasoning(self, prompt: str) -> Optional[str]:
        """
        调用 LLM 生成推理文本

        Args:
            prompt: 输入 prompt

        Returns:
            生成的推理文本，失败返回 None
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a systems biology expert specializing in cellular signaling pathways and gene regulation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # 尝试从 content 字段获取
            reasoning = response.choices[0].message.content

            # 如果 content 为 None，尝试从 reasoning 字段获取（Qwen reasoning models）
            if reasoning is None:
                reasoning_raw = getattr(response.choices[0].message, 'reasoning', None)
                if reasoning_raw:
                    # Qwen 推理模型的 reasoning 包含思考过程 + 实际答案
                    # 尝试提取引号中的内容（实际答案通常在引号里）
                    import re
                    # 查找最后一对引号中的内容
                    quoted = re.findall(r'"([^"]+)"', reasoning_raw)
                    if quoted:
                        # 使用最长的引用内容（通常是实际答案）
                        reasoning = max(quoted, key=len)
                    else:
                        # 如果没有引号，使用整个 reasoning
                        reasoning = reasoning_raw
                else:
                    reasoning = None

            if reasoning is None:
                logger.warning("Both content and reasoning fields are None")
                return None

            return reasoning.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def filter_reasoning(self, reasoning: str, required_entities: List[str]) -> bool:
        """
        质量过滤

        Args:
            reasoning: 生成的推理文本
            required_entities: 必需的实体列表

        Returns:
            是否通过过滤
        """
        if not reasoning:
            return False

        # 长度检查 - 只要有内容就接受（推理模型输出变化大）
        word_count = len(reasoning.split())
        if word_count < 5:  # 只过滤掉极短的
            logger.debug(f"Filtering: length {word_count} < 5")
            return False

        # 关键词黑名单
        reasoning_lower = reasoning.lower()
        for keyword in self.blacklist_keywords:
            if keyword in reasoning_lower:
                logger.debug(f"Filtering: contains blacklist keyword '{keyword}'")
                return False

        # 实体完整性检查 - 只检查扰动或目标基因其中之一
        # 推理模型的输出可能不会明确提到所有实体
        has_pert = any(entity.lower() in reasoning.lower() for entity in required_entities[:1])
        has_gene = any(entity.lower() in reasoning.lower() for entity in required_entities[1:2]) if len(required_entities) > 1 else True

        if not (has_pert or has_gene):
            logger.debug(f"Filtering: neither perturbation nor gene mentioned")
            return False

        return True

    def process_sample(self, sample: Dict) -> Optional[Dict]:
        """
        处理单个样本，生成推理文本

        Args:
            sample: 带路径的样本

        Returns:
            带推理文本的样本，失败返回 None
        """
        if not sample.get('reasoning_paths'):
            logger.debug(f"Sample {sample['id']} has no paths")
            return None

        perturbation = sample['input']['pert']
        target_gene = sample['input']['gene']
        label = sample['input']['label']
        basal_context = sample['basal_context']

        # 选择第一条路径（最短路径）
        path_data = sample['reasoning_paths'][0]
        triplets = path_data['triplets']

        # 构建 prompt
        prompt = self.build_prompt(
            perturbation,
            target_gene,
            label,
            triplets,
            basal_context
        )

        # 生成推理
        reasoning = self.generate_reasoning(prompt)

        if not reasoning:
            logger.debug(f"Failed to generate reasoning for {sample['id']}")
            return None

        # 提取必需实体
        required_entities = [perturbation, target_gene]
        for source, _, target in triplets:
            required_entities.extend([source, target])
        required_entities = list(set(required_entities))

        # 质量过滤
        if not self.filter_reasoning(reasoning, required_entities):
            logger.debug(f"Reasoning filtered out for {sample['id']}")
            return None

        # 添加到样本
        sample['output'] = {
            'reasoning_text': reasoning,
            'reasoning_id': 0,
            'triplets_ref': triplets,
            'path_length': path_data['length']
        }

        return sample

    def batch_process(self, samples: List[Dict], batch_size: int = 1) -> List[Dict]:
        """
        批量处理样本

        Args:
            samples: 带路径的样本列表
            batch_size: 批次大小（当前实现为逐个处理）

        Returns:
            带推理文本的样本列表
        """
        logger.info(f"Generating reasoning text for {len(samples)} samples...")

        results = []
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Generated {i}/{len(samples)} reasonings")

            try:
                result = self.process_sample(sample)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process sample {sample['id']}: {e}")
                continue

        logger.info(f"LLM generation completed: {len(results)}/{len(samples)} samples")
        return results
