"""
Dialogue Summarization Studio
================================
A professional Streamlit interface for a FLAN-T5 model fine-tuned with LoRA
on the SAMSum dialogue-summarization dataset.

Expected artifact folder (default: ./lora_model) containing the files
exported from the training notebook:
    lora_config / adapter_config.json
    lora_model.safetensors / adapter_model.safetensors
    tokenizer, tokenizer_config, special_tokens_map, spiece.model
    (training_args.bin, optimizer.pt, etc. are ignored at inference time)

Run with:
    streamlit run app.py
"""

import os
import json
import time
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, LoraConfig, get_peft_model


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dialogue Summarization Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_MODEL_NAME = "google/flan-t5-base"
FT_PROMPT_PREFIX = "Summarize the following conversation:\n"
MAX_INPUT_LENGTH = 1024

# Benchmark results captured from the training notebook (Section 9 / 10)
BENCHMARK_RESULTS = [
    {"method": "Zero-shot",  "rouge1": 0.4966, "rouge2": 0.2393, "rougeL": 0.4080, "bertscore_f1": 0.9139, "avg_latency_sec": 0.597, "trainable_params_pct": 0.0,    "train_time_min": 0.0},
    {"method": "One-shot",   "rouge1": 0.4946, "rouge2": 0.2304, "rougeL": 0.4039, "bertscore_f1": 0.9136, "avg_latency_sec": 0.635, "trainable_params_pct": 0.0,    "train_time_min": 0.0},
    {"method": "Few-shot",   "rouge1": 0.2073, "rouge2": 0.0501, "rougeL": 0.1809, "bertscore_f1": 0.8759, "avg_latency_sec": 0.452, "trainable_params_pct": 0.0,    "train_time_min": 0.0},
    {"method": "LoRA",       "rouge1": 0.4952, "rouge2": 0.2304, "rougeL": 0.3639, "bertscore_f1": 0.9153, "avg_latency_sec": 1.007, "trainable_params_pct": 0.710, "train_time_min": 95.0},
    {"method": "Adapter",    "rouge1": 0.4594, "rouge2": 0.1992, "rougeL": 0.3588, "bertscore_f1": 0.9113, "avg_latency_sec": 1.070, "trainable_params_pct": 0.717, "train_time_min": 89.7},
]

EXAMPLE_DIALOGUES = {
    "Assignment deadline": """Sara: Hey Ali, have you finished the machine learning assignment?
Ali: Not yet. I completed the data preprocessing part, but I'm still working on the model evaluation.
Sara: The submission deadline is tomorrow at 5 PM.
Ali: I know. I plan to finish it tonight and double-check the results.
Sara: Great. Don't forget to include the confusion matrix and ROC curve in your report.
Ali: Thanks for reminding me. I'll also upload the notebook to GitHub before submitting.
Sara: Sounds good. Let me know if you need any help.
Ali: Will do. Thanks!""",
    "Weekend plans": """Jake: Are we still on for hiking this Saturday?
Mia: Yes! I checked the weather, it's supposed to be sunny.
Jake: Perfect. Should we do the Blue Ridge trail again or try something new?
Mia: Let's try the new one near the lake, I heard the view is amazing.
Jake: Sounds good. I'll bring the snacks, you bring the water?
Mia: Deal. Let's leave by 7 AM to beat the crowds.
Jake: Works for me. See you then!""",
    "Office scheduling": """Priya: Can we move tomorrow's 10 AM meeting to 2 PM?
Tom: I have a call at 2, but I'm free at 3.
Priya: 3 works. I'll send a new invite.
Tom: Thanks. Should I still prep the slides for the budget review?
Priya: Yes, please, and add the Q3 numbers we discussed.
Tom: Got it, I'll have them ready by tomorrow morning.""",
}


# ----------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0rem;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }
    .summary-box {
        background-color: #F5F3FF;
        border-left: 5px solid #7C3AED;
        padding: 1.2rem 1.4rem;
        border-radius: 0.5rem;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .metric-card {
        background-color: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Model loading (cached so it only happens once per session)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(adapter_path: str):
    """Loads the FLAN-T5 base model and applies the trained LoRA adapter.

    Supports two artifact layouts:
      1. Standard PEFT save (adapter_config.json + adapter_model.safetensors)
         -> loaded directly with PeftModel.from_pretrained
      2. Custom naming used by this notebook's export
         (lora_config / lora_model.safetensors)
         -> config + weights are loaded manually and applied with get_peft_model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_source = adapter_path if _has_tokenizer_files(adapter_path) else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    adapter_dir = Path(adapter_path)
    standard_config = adapter_dir / "adapter_config.json"
    standard_weights_st = adapter_dir / "adapter_model.safetensors"
    standard_weights_bin = adapter_dir / "adapter_model.bin"

    custom_config_candidates = [adapter_dir / "lora_config.json", adapter_dir / "lora_config"]
    custom_weights = adapter_dir / "lora_model.safetensors"

    model = None
    load_mode = None

    if standard_config.exists() and (standard_weights_st.exists() or standard_weights_bin.exists()):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        load_mode = "standard PEFT adapter"
    else:
        config_path = next((p for p in custom_config_candidates if p.exists()), None)
        if config_path is not None and custom_weights.exists():
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
            lora_cfg = LoraConfig(
                task_type=cfg_dict.get("task_type", "SEQ_2_SEQ_LM"),
                r=cfg_dict.get("r", 16),
                lora_alpha=cfg_dict.get("lora_alpha", 32),
                lora_dropout=cfg_dict.get("lora_dropout", 0.05),
                target_modules=cfg_dict.get("target_modules", ["q", "v"]),
                bias=cfg_dict.get("bias", "none"),
            )
            model = get_peft_model(base_model, lora_cfg)

            from safetensors.torch import load_file
            state_dict = load_file(str(custom_weights))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            load_mode = "custom LoRA export"
        else:
            model = base_model
            load_mode = "base model only (no adapter found)"

    model.to(device)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "load_mode": load_mode,
        "trainable_params": trainable,
        "total_params": total,
    }


def _has_tokenizer_files(path: str) -> bool:
    p = Path(path)
    return (p / "tokenizer_config.json").exists() or (p / "spiece.model").exists()


def generate_summary(bundle, dialogue_text, max_new_tokens, num_beams, do_sample, temperature):
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]

    prompt = FT_PROMPT_PREFIX + dialogue_text
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, num_beams=1)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    latency = time.time() - t0

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary, latency


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:

    st.markdown("---")
    st.markdown("### 🎛️ Generation settings")
    max_new_tokens = st.slider("Max summary length (tokens)", 20, 200, 100, step=10)
    num_beams = st.slider("Beam search width", 1, 8, 4)
    do_sample = st.toggle("Enable sampling (creative mode)", value=False)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1, disabled=not do_sample)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "Base model: **google/flan-t5-base**\n\n"
        "Fine-tuning: **LoRA** (r=16, α=32, target modules: q, v)\n\n"
        "Dataset: **SAMSum** dialogue corpus"
    )

# Load model (cached)
load_error = None
bundle = None
try:
    with st.spinner("Loading model and adapter weights..."):
        bundle = load_model(adapter_path)
except Exception as e:
    load_error = str(e)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="main-header">💬 Dialogue Summarization Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">FLAN-T5 + LoRA fine-tuned on the SAMSum corpus — turn multi-turn chats into concise summaries.</div>',
    unsafe_allow_html=True,
)

if load_error:
    st.error(
        f"⚠️ Could not load the model from `{adapter_path}`.\n\n**Details:** {load_error}\n\n"
        "Double-check that the folder path is correct and contains the exported adapter files. "
        "The app will still let you inspect the benchmark results below."
    )

status_cols = st.columns(4)
if bundle:
    with status_cols[0]:
        st.markdown(f'<div class="metric-card"><b>Device</b><br>{bundle["device"].upper()}</div>', unsafe_allow_html=True)
    with status_cols[1]:
        st.markdown(f'<div class="metric-card"><b>Load mode</b><br>{bundle["load_mode"]}</div>', unsafe_allow_html=True)
    with status_cols[2]:
        pct = 100 * bundle["trainable_params"] / bundle["total_params"] if bundle["total_params"] else 0
        st.markdown(f'<div class="metric-card"><b>Trainable params</b><br>{pct:.2f}%</div>', unsafe_allow_html=True)
    with status_cols[3]:
        st.markdown(f'<div class="metric-card"><b>Total params</b><br>{bundle["total_params"]/1e6:.1f}M</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_summarize, tab_benchmarks, tab_history = st.tabs(["📝 Summarize", "📊 Benchmarks", "🕘 History"])

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Summarize tab ----------------
with tab_summarize:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("#### Conversation input")
        example_choice = st.selectbox(
            "Try a sample dialogue (optional)",
            ["— write your own —"] + list(EXAMPLE_DIALOGUES.keys()),
        )
        default_text = "" if example_choice == "— write your own —" else EXAMPLE_DIALOGUES[example_choice]

        dialogue_input = st.text_area(
            "Paste or type a dialogue",
            value=default_text,
            height=280,
            placeholder="Alice: Hey, are you free tonight?\nBob: Yeah, what's up?\n...",
        )

        generate_clicked = st.button("✨ Generate summary", use_container_width=True, disabled=bundle is None)

    with right:
        st.markdown("#### Summary")
        if generate_clicked:
            if not dialogue_input.strip():
                st.warning("Please enter a conversation to summarize.")
            else:
                with st.spinner("Generating summary..."):
                    summary, latency = generate_summary(
                        bundle, dialogue_input, max_new_tokens, num_beams, do_sample, temperature
                    )
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                st.caption(f"⏱️ Generated in {latency:.2f}s  •  {num_beams} beams  •  max {max_new_tokens} tokens")

                st.session_state.history.insert(0, {
                    "dialogue": dialogue_input,
                    "summary": summary,
                    "latency": latency,
                })
        else:
            st.info("Enter a conversation on the left and click **Generate summary**.")

# ---------------- Benchmarks tab ----------------
with tab_benchmarks:
    st.markdown("#### Method comparison (from training notebook, SAMSum test set)")

    import pandas as pd
    df = pd.DataFrame(BENCHMARK_RESULTS)
    display_df = df.rename(columns={
        "method": "Method", "rouge1": "ROUGE-1", "rouge2": "ROUGE-2", "rougeL": "ROUGE-L",
        "bertscore_f1": "BERTScore F1", "avg_latency_sec": "Latency (s)",
        "trainable_params_pct": "Trainable params (%)", "train_time_min": "Train time (min)",
    })
    st.dataframe(display_df.style.format({
        "ROUGE-1": "{:.4f}", "ROUGE-2": "{:.4f}", "ROUGE-L": "{:.4f}",
        "BERTScore F1": "{:.4f}", "Latency (s)": "{:.3f}",
        "Trainable params (%)": "{:.3f}", "Train time (min)": "{:.1f}",
    }), use_container_width=True, hide_index=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("**ROUGE-L by method**")
        st.bar_chart(df.set_index("method")["rougeL"])
    with chart_cols[1]:
        st.markdown("**BERTScore F1 by method**")
        st.bar_chart(df.set_index("method")["bertscore_f1"])

    st.info(
        "**Takeaway:** LoRA edged out Adapter fine-tuning across every metric at a matched "
        "trainable-parameter budget (~0.71%). Neither PEFT method surpassed FLAN-T5's strong "
        "zero-shot baseline on SAMSum — likely because the base model's instruction-tuning "
        "already covers summarization-style tasks well. LoRA was selected as the deployed model "
        "for its consistent edge over Adapters."
    )

# ---------------- History tab ----------------
with tab_history:
    st.markdown("#### Past summaries this session")
    if not st.session_state.history:
        st.caption("No summaries generated yet.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"Summary #{len(st.session_state.history) - i}  •  {item['latency']:.2f}s"):
                st.markdown("**Dialogue:**")
                st.text(item["dialogue"])
                st.markdown("**Summary:**")
                st.markdown(f'<div class="summary-box">{item["summary"]}</div>', unsafe_allow_html=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
