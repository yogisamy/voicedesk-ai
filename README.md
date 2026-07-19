# 🎧 VoiceDesk AI — API-driven Customer Support Assistant

**CCZG506 Assignment II** · Domain: **Customer Service** · Categories: **NLP + Speech Recognition**

A customer records (or types) a complaint. The system transcribes it, detects sentiment,
classifies intent (zero-shot **and** with our fine-tuned model), extracts entities,
summarises the ticket for a human agent, drafts an empathetic reply with an LLM, and
speaks the reply back — while logging LLMOps metrics for every API call.

## Architecture

```
🎙️ Voice / Text complaint
        │
        ▼
[1] Whisper-large-v3 (HF API) ── Speech-to-Text          (Speech Recognition)
        │
        ├─▶ [3] DistilBERT-SST-2 ─ Sentiment              (NLP)
        ├─▶ [4] BART-MNLI zero-shot ┐
        │       DistilBERT fine-tuned ┘ Intent            (NLP)  ← requirement #8
        ├─▶ [5] BERT-NER ─ Named Entities                 (NLP)
        ├─▶ [6] BART-CNN ─ Ticket Summary                 (NLP)
        └─▶ [7] Qwen2.5-7B-Instruct (HF API) ─ Reply      (NLP)
                    │
                    ▼
        [2] gTTS ─ Text-to-Speech reply                   (Speech Recognition)

All calls ──▶ metrics_log.csv ──▶ 📊 LLMOps Dashboard tab
```

## Setup (5 minutes)

1. Get a free Hugging Face token: https://huggingface.co/settings/tokens (Read access).
2. ```bash
   pip3 install -r requirements.txt
   export HF_TOKEN=hf_xxxxxxxxxxxx      # Windows: set HF_TOKEN=hf_xxx
   python3 app.py
   ```
3. Open the local Gradio URL it prints.

## Fine-tuning (requirement #8)

Open Google Colab (free T4 GPU), upload `finetune_intent.py`, then:

```
!pip install -q transformers datasets evaluate accelerate scikit-learn
!python finetune_intent.py
!zip -r intent-model.zip intent-model
```

Download `intent-model.zip`, unzip it **next to `app.py`**. The app now shows
zero-shot vs fine-tuned intent predictions side by side — a strong demo/viva point.
The script prints test **accuracy** and **macro-F1**; quote them in the report.

- Base model: DistilBERT (SLM, 66M params)
- Dataset: Bitext Customer Support (~27k utterances, 27 intents) — same domain ✔

## LLMOps — metrics measured (≥5 required)

| Metric | Where |
|---|---|
| 1. Latency per sub-task | every API call, logged to CSV |
| 2. Model confidence (sentiment / intent / NER) | logged per call |
| 3. Intent quality: fine-tuned accuracy & macro-F1 vs zero-shot | finetune script + live comparison |
| 4. Summary quality (ROUGE-1 F1) | logged per summary |
| 5. Token usage & estimated cost | logged per LLM call |
| 6. User satisfaction (👍/👎 feedback) | in-app rating |

View them in the **📊 LLMOps Dashboard** tab (aggregated from `metrics_log.csv`).

## Suggested demo flow (viva)

1. Record: *"Hi, this is Priya. My order 4521 from Chennai arrived damaged and I want a refund."*
2. Show transcript → sentiment (NEGATIVE) → intents (both models agree: refund) →
   entities (Priya=PER, Chennai=LOC) → summary → generated reply → play TTS audio.
3. Rate the reply 👍, open the Dashboard tab, hit refresh → show live metrics.
4. Explain the fine-tuning notebook output (accuracy/F1) and why an SLM was chosen.

## Rubric mapping

| Assignment requirement | Where satisfied |
|---|---|
| Domain | Customer Service |
| Two categories | NLP + Speech Recognition |
| ≥5 sub-tasks | 7 sub-tasks (see architecture) |
| LLM/SLM models via APIs | Hugging Face Inference API (Whisper, DistilBERT, BART, Qwen2.5) + gTTS |
| Cohesive unified objective | One end-to-end support-ticket pipeline |
| Interactive & demonstrable | Gradio web app |
| LLMOps, ≥5 metrics | 6 metrics + dashboard |
| Fine-tune on same-domain dataset | DistilBERT on Bitext Customer Support |
