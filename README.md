# Text-Summarization-Using-LoRA-and-Adapter-Layers
# Dialogue Summarization Studio — LoRA vs. Adapters vs. Prompting on FLAN-T5

A full experimental pipeline comparing **prompting-based** and **parameter-efficient fine-tuning (PEFT)** methods for abstractive dialogue summarization, plus a deployable **Streamlit** app serving the best-performing model.

Built on `google/flan-t5-large` and the [SAMSum Corpus](https://huggingface.co/datasets/knkarthick/samsum) (Gliwa et al., 2019), this project benchmarks five approaches head-to-head — **Zero-shot, One-shot, Few-shot, LoRA, and Adapter layers** — across accuracy, efficiency, and latency, then ships the winning model in a production-style inference UI.

---

## What's Inside

| Component | Description |
|---|---|
| `text-summarizartion-using-lora-and-adapters.ipynb` | End-to-end notebook: data exploration → prompting baselines → LoRA fine-tuning → Adapter fine-tuning → benchmarking → model export |
| `app.py` | Streamlit inference app that loads the exported LoRA adapter and generates summaries interactively |
| `requirements.txt` | Python dependencies |
| `lora_model` | Essential model files |

---

## Methodology

Run on a Kaggle notebook with **2× NVIDIA T4 GPUs**, the pipeline evaluates:

1. **Zero-shot prompting** — instruction-only prompting of the base FLAN-T5-large model, no examples.
2. **One-shot in-context learning** — one dialogue→summary example prepended to the prompt.
3. **Few-shot in-context learning** — multiple examples prepended, adapted for FLAN-T5's smaller context window relative to decoder-only LLMs.
4. **LoRA fine-tuning** — low-rank adapters injected into the attention projection layers via Hugging Face `peft`, base weights frozen.
5. **Adapter-layer fine-tuning** — bottleneck adapter modules inserted via the AdapterHub `adapters` library, base weights frozen.

Every method is scored on the same held-out SAMSum test set using:

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — lexical overlap with reference summaries
- **BERTScore F1** — semantic similarity, catching valid paraphrases ROUGE misses
- **Trainable parameters (%)** — fine-tuning efficiency
- **Training time** — wall-clock cost
- **Inference latency** — per-sample generation time

## Results

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Trainable Params (%) | Train Time (min) |
|---|---|---|---|---|---|---|
| Zero-shot | 0.4966 | 0.2393 | 0.4080 | 0.9139 | 0.0 | 0.0 |
| One-shot | 0.4946 | 0.2304 | 0.4039 | 0.9136 | 0.0 | 0.0 |
| Few-shot | 0.2073 | 0.0501 | 0.1809 | 0.8759 | 0.0 | 0.0 |
| **LoRA** | 0.4952 | 0.2304 | 0.3639 | **0.9153** | 0.710 | 95.0 |
| Adapter | 0.4594 | 0.1992 | 0.3588 | 0.9113 | 0.717 | 89.7 |

**Key finding:** LoRA outperformed Adapter fine-tuning across every metric at a matched trainable-parameter budget (~0.71%). Neither PEFT method surpassed FLAN-T5's strong zero-shot baseline — likely because the base model's instruction-tuning already covers summarization-style tasks well. **LoRA was selected as the deployed model** for its consistent edge over Adapters at comparable cost.

---

## Streamlit App

The app (`app.py`) loads a FLAN-T5 base model and applies a trained LoRA adapter for interactive dialogue summarization.

**Features**
- Paste any multi-turn dialogue or pick from built-in sample conversations
- Adjustable generation settings: max length, beam width, sampling/temperature
- Live benchmark tab visualizing the comparison table above
- Session history of past summaries
- Supports both standard PEFT adapter exports (`adapter_config.json` + `adapter_model.safetensors`) and the notebook's custom export format (`lora_config` + `lora_model.safetensors`)

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default the app looks for adapter files in `./lora_model`. Update the **LoRA artifact folder** field in the sidebar to point to your exported checkpoint, or edit `BASE_MODEL_NAME` / the default path in `app.py` to match the model size used during training.

---

## Requirements

```
streamlit>=1.36
torch>=2.1
transformers>=4.40
peft>=0.11
safetensors>=0.4
pandas>=2.0
sentencepiece>=0.1.99
```

---

## Repository Structure

```
.
├── text-summarizartion-using-lora-and-adapters.ipynb   # Training & benchmarking notebook
├── app.py                                              # Streamlit inference app
├── requirements.txt                                    # Dependencies
├── lora_model                                          # model files
└── README.md
```

---

## Future Improvements

- Export and benchmark on larger FLAN-T5 variants (XL)
- Add quantized (4-bit/8-bit) inference for lower-latency deployment
- Extend the comparison to additional PEFT methods (Prefix Tuning, IA³)
- Add automated adapter-format detection tests and CI

---

## References

- Gliwa, B. et al. (2019). *SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization.*
- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
- Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP.* (Adapter layers)
- Chung, H. W. et al. (2022). *Scaling Instruction-Finetuned Language Models.* (FLAN-T5)

---

## License

This project is released for educational and research purposes. Check individual dataset/model licenses (SAMSum, FLAN-T5) before commercial use.
