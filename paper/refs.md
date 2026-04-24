# Reference list for *Norm Separation is Necessary* (preliminary)

To be converted into bibtex when the arxiv submission is prepared.
All entries below are the ones already mentioned in the current
`nsn.md` draft. Venue / arXiv IDs should be verified before
submission.

---

## KV cache quantization (primary related work)

- **KIVI** — Liu, Z., et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." *ICML 2024*. arXiv:2402.02750.
- **KVQuant** — Hooper, C., et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." *NeurIPS 2024*. https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf
- **Atom** — Zhao, Y., et al. "Atom: Low-bit Quantization for Efficient and Accurate LLM Serving." *MLSys 2024*. arXiv:2310.19102.
- **PolarQuant** — Wang, et al. "PolarQuant: Efficient KV Cache Quantization with Polar Decomposition." 2025. arXiv:2502.02617.
- **TurboQuant** — (Google Research). "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *ICLR 2026*. arXiv:2504.19874. The paper uses a random rotation, a Lloyd-Max scalar quantizer on the rotated coordinates, and a QJL-style residual correction; our earlier draft titled it by mechanism rather than by the published title. Correction flagged in the GPT-5 peer-review pass on v0.3.
- **KVLinC** — (preprint). "KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction." October 2025. arXiv:2510.05373.
- **SQuat** — (2025). "SQuat: Subspace-Orthogonal KV Cache Quantization." arXiv:2503.24358.

## Weight and activation quantization (ingredients transplanted in the ablation)

- **LLM.int8** — Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *NeurIPS 2022*.
- **SmoothQuant** — Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML 2023*. arXiv:2211.10438.
- **GPTQ** — Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*.
- **AWQ** — Lin, J., Tang, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." *MLSys 2024*.
- **SpQR** — Dettmers, T., et al. "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression." *ICLR 2024*.

## Rotation-based quantization

- **QuaRot** — Ashkboos, S., et al. "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs." *NeurIPS 2024*.
- **SpinQuant** — (2024). "SpinQuant: LLM Quantization with Learned Rotations." arXiv:2405.16406.

## Companion paper (same author)

- **Sato, K.** "Norm-Separated Quantization: A Training-Free Fix for KV Cache INT4 Failures." 2026. *Zenodo preprint.* https://github.com/metaSATOKEN/norm-separated-quantization

## Benchmarks

- **WikiText-2** — Merity, S., Xiong, C., Bradbury, J., Socher, R. "Pointer Sentinel Mixture Models." *ICLR 2017*. arXiv:1609.07843.

## Models benchmarked in this paper

(Not citations in the usual sense; we cite the model cards or papers on arxiv submission.)

- GPT-J-6B (EleutherAI, 2021)
- Pythia-6.9B (Biderman et al., 2023)
- Mistral-7B-v0.1 (Jiang et al., 2023)
- Qwen2-7B (Alibaba, 2024)
- Qwen2.5-14B (Alibaba, 2024)
- Llama-3.1-8B (Meta, 2024)
- Qwen3.5-9B (Alibaba, 2026-02)
- Gemma-4-E4B (Google DeepMind, 2026-04)
- Phi-3-medium-4k-instruct (Microsoft, 2024) — attempted, blocked
