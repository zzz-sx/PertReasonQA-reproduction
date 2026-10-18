"""
配置文件
定义数据路径、模型参数等
"""

from pathlib import Path

# ============================================
# 路径配置
# ============================================

# 项目根目录
PROJECT_ROOT = Path("/cpfs01/sihan.yang/PertReasonQA")

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
ORIGIN_DATA_DIR = DATA_DIR / "origin_data"
MSIGDB_DIR = DATA_DIR / "msigdb/msigdb_v2026.1.Hs_files_to_download_locally/msigdb_v2026.1.Hs_GMTs"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "annotations"

# 知识图谱文件
HALLMARK_GMT = MSIGDB_DIR / "h.all.v2026.1.Hs.symbols.gmt"
KEGG_GMT = MSIGDB_DIR / "c2.cp.kegg_medicus.v2026.1.Hs.symbols.gmt"
REACTOME_GMT = MSIGDB_DIR / "c2.cp.reactome.v2026.1.Hs.symbols.gmt"

# ============================================
# 数据处理参数
# ============================================

# QC 参数
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3

# 差异表达参数
DE_P_VALUE_THRESHOLD_UP = 0.01      # p < 0.01 显著
DE_P_VALUE_THRESHOLD_DOWN = 0.1     # p > 0.1 不显著
DE_LOG2FC_THRESHOLD = 0.5           # log2FC 阈值

# KS 阈值参数
KS_ALPHA = 0.05

# 路径提取参数
MAX_PATH_LENGTH = 5
MAX_PATHS_PER_GENE = 10
PAGERANK_ALPHA = 0.85

# ============================================
# LLM 参数
# ============================================

# vLLM server 默认配置
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096  # 增加到 4096 以适应推理模型的长输出

# 生成参数
MIN_REASONING_LENGTH = 20  # 最少单词数
MAX_REASONING_LENGTH = 200  # 最多单词数

# 质量过滤关键词黑名单
BLACKLIST_KEYWORDS = [
    'unclear',
    'uncertain',
    'may or may not',
    'not enough information',
    'cannot determine',
    'i don\'t know',
    'i am not sure'
]

# ============================================
# 采样参数
# ============================================

DEFAULT_N_SAMPLES = 1000
DEFAULT_SEED = 42

# ============================================
# 输出格式
# ============================================

OUTPUT_JSONL_FILENAME = "train_reasoning_{n_samples}_seed_{seed}.jsonl"
