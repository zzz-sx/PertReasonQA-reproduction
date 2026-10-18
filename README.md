# PertReasonQA Stage3 Reproduction Archive

This archive preserves the cloud-side artifacts from the PertReasonQA
training-set construction reproduction.

## Main result

Full Stage3 run:
- Input Stage2 retained samples: 3668
- Stage3 saved samples: 3668
- First-pass failures: 0

Main output:
`pertreason_stage3/data/annotations/kg_qwen25_bf16_stage3_full3668.jsonl`

Main output SHA256:
`35c1a19e6e77fc18ed61039e5dd6acb8bc7b5b7aac7696b2dce01f6e032442b5`

Final portable backup SHA256:
`1f6222dcb8f3252e9c5596bd34e97d2f6f14672b641a9d3e650fb4a986fa71ff`

## Runtime

Original PertReasonQA Stage3 generation logic was used with:
- Qwen/Qwen2.5-7B-Instruct
- BF16
- local vLLM OpenAI-compatible server
- NVIDIA A10 24GB
- resume/checkpoint wrapper only for execution robustness

The wrapper does not replace the repository Stage3 generation algorithm.

See:
- `ENVIRONMENT.txt`
- `pip-freeze.txt`
- `nvidia-smi.txt`
- `SHA256SUMS.txt`

The downloaded Qwen model weights are intentionally not committed because
they are approximately 15GB and can be reacquired from ModelScope.
