"""
VoiceDesk AI — API-driven AI Customer Support Assistant
CCZG506 Assignment II | Domain: Customer Service
Categories: NLP + Speech Recognition

Sub-tasks implemented (7):
  1. Speech-to-Text        (Speech)  - openai/whisper-large-v3 via HF Inference API
  2. Text-to-Speech        (Speech)  - gTTS API
  3. Sentiment Analysis    (NLP)     - distilbert/distilbert-base-uncased-finetuned-sst-2-english
  4. Intent Classification (NLP)     - zero-shot bart-large-mnli  +  our FINE-TUNED DistilBERT
  5. Named Entity Recognition (NLP)  - dslim/bert-base-NER
  6. Summarization         (NLP)     - facebook/bart-large-cnn
  7. Response Generation   (NLP)     - LLM chat model via HF Inference API

LLMOps metrics logged to metrics_log.csv and visualised in the Dashboard tab:
  latency per sub-task | sentiment confidence | intent confidence (zero-shot vs fine-tuned)
  ROUGE-1 summary quality | token usage & est. cost | user feedback (thumbs up/down)

Run:
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxx        # free token from huggingface.co/settings/tokens
  python app.py
"""

import os
import time
import csv
import datetime
import tempfile

import gradio as gr
import pandas as pd
from huggingface_hub import InferenceClient
from gtts import gTTS
from rouge_score import rouge_scorer

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = InferenceClient(token=HF_TOKEN)

ASR_MODEL = "openai/whisper-large-v3"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ZEROSHOT_MODEL = "facebook/bart-large-mnli"
NER_MODEL = "dslim/bert-base-NER"
SUMMARY_MODEL = "facebook/bart-large-cnn"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # any chat model listed at router.huggingface.co/v1/models

FINETUNED_DIR = "./intent-model"     # produced by finetune_intent.py
METRICS_FILE = "metrics_log.csv"

CANDIDATE_INTENTS = [
    "refund request", "cancel order", "delivery problem",
    "payment issue", "product complaint", "account help",
    "general enquiry",
]

EST_COST_PER_1K_TOKENS = 0.0002  # illustrative cost for LLMOps cost tracking

# ----------------------------------------------------------------------------
# LLMOps: metrics logger
# ----------------------------------------------------------------------------
METRIC_COLUMNS = [
    "timestamp", "subtask", "model", "latency_sec",
    "confidence", "tokens", "est_cost_usd", "extra",
]


def log_metric(subtask, model, latency, confidence=None, tokens=None, extra=""):
    new_file = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(METRIC_COLUMNS)
        cost = round(tokens / 1000 * EST_COST_PER_1K_TOKENS, 6) if tokens else ""
        w.writerow([
            datetime.datetime.now().isoformat(timespec="seconds"),
            subtask, model, round(latency, 3),
            round(confidence, 4) if confidence is not None else "",
            tokens or "", cost, extra,
        ])


def timed(subtask, model_name, fn, *args, **kwargs):
    """Run fn, measure latency, return (result, latency)."""
    t0 = time.time()
    result = fn(*args, **kwargs)
    latency = time.time() - t0
    return result, latency


# ----------------------------------------------------------------------------
# Sub-task 1: Speech-to-Text (Speech Recognition)
# ----------------------------------------------------------------------------
def speech_to_text(audio_path):
    if not audio_path:
        return ""
    result, latency = timed("ASR", ASR_MODEL,
                            client.automatic_speech_recognition,
                            audio_path, model=ASR_MODEL)
    text = result.text if hasattr(result, "text") else str(result)
    log_metric("speech_to_text", ASR_MODEL, latency,
               tokens=len(text.split()), extra=f"chars={len(text)}")
    return text.strip()


# ----------------------------------------------------------------------------
# Sub-task 2: Text-to-Speech (Speech Recognition category)
# ----------------------------------------------------------------------------
def text_to_speech(text):
    if not text:
        return None
    t0 = time.time()
    tts = gTTS(text=text[:500], lang="en")
    out_path = os.path.join(tempfile.gettempdir(), "voicedesk_reply.mp3")
    tts.save(out_path)
    log_metric("text_to_speech", "gTTS", time.time() - t0,
               tokens=len(text.split()))
    return out_path


# ----------------------------------------------------------------------------
# Sub-task 3: Sentiment Analysis (NLP)
# ----------------------------------------------------------------------------
def analyze_sentiment(text):
    result, latency = timed("sentiment", SENTIMENT_MODEL,
                            client.text_classification, text,
                            model=SENTIMENT_MODEL)
    top = result[0]
    log_metric("sentiment", SENTIMENT_MODEL, latency, confidence=top.score)
    return f"{top.label} ({top.score:.2%})", top.label, top.score


# ----------------------------------------------------------------------------
# Sub-task 4a: Intent Classification — zero-shot baseline (NLP)
# ----------------------------------------------------------------------------
def classify_intent_zeroshot(text):
    result, latency = timed("intent_zeroshot", ZEROSHOT_MODEL,
                            client.zero_shot_classification, text,
                            candidate_labels=CANDIDATE_INTENTS,
                            model=ZEROSHOT_MODEL)
    top = result[0]
    log_metric("intent_zeroshot", ZEROSHOT_MODEL, latency, confidence=top.score)
    return top.label, top.score


# ----------------------------------------------------------------------------
# Sub-task 4b: Intent Classification — OUR FINE-TUNED MODEL (NLP)
# ----------------------------------------------------------------------------
_finetuned_pipe = None


def classify_intent_finetuned(text):
    """Loads the locally fine-tuned DistilBERT (see finetune_intent.py)."""
    global _finetuned_pipe
    if not os.path.isdir(FINETUNED_DIR):
        return "fine-tuned model not found (run finetune_intent.py)", 0.0
    if _finetuned_pipe is None:
        from transformers import pipeline
        _finetuned_pipe = pipeline("text-classification", model=FINETUNED_DIR)
    t0 = time.time()
    out = _finetuned_pipe(text[:512])[0]
    latency = time.time() - t0
    log_metric("intent_finetuned", "distilbert-finetuned(Bitext)",
               latency, confidence=out["score"])
    return out["label"], out["score"]


# ----------------------------------------------------------------------------
# Sub-task 5: Named Entity Recognition (NLP)
# ----------------------------------------------------------------------------
def extract_entities(text):
    result, latency = timed("ner", NER_MODEL,
                            client.token_classification, text,
                            model=NER_MODEL)
    ents = [f"{e.word} ({e.entity_group})" for e in result]
    avg_conf = sum(e.score for e in result) / len(result) if result else 0
    log_metric("ner", NER_MODEL, latency, confidence=avg_conf,
               extra=f"entities={len(ents)}")
    return ", ".join(ents) if ents else "No named entities found"


# ----------------------------------------------------------------------------
# Sub-task 6: Summarization (NLP) + ROUGE quality metric
# ----------------------------------------------------------------------------
_rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)


def summarize(text):
    if len(text.split()) < 15:
        return text  # too short to summarise
    result, latency = timed("summarization", SUMMARY_MODEL,
                            client.summarization, text, model=SUMMARY_MODEL)
    summary = result.summary_text if hasattr(result, "summary_text") else str(result)
    rouge1 = _rouge.score(text, summary)["rouge1"].fmeasure
    log_metric("summarization", SUMMARY_MODEL, latency,
               confidence=rouge1, tokens=len(summary.split()),
               extra="confidence column = ROUGE-1 F1")
    return summary


# ----------------------------------------------------------------------------
# Sub-task 7: Response Generation (NLP, LLM via API)
# ----------------------------------------------------------------------------
def generate_response(complaint, sentiment_label, intent):
    prompt = (
        "You are a polite customer-support agent for an online store. "
        f"The customer's message (sentiment: {sentiment_label}, "
        f"detected intent: {intent}) is:\n\n{complaint}\n\n"
        "Write a short, empathetic, professional reply (max 120 words) "
        "with a concrete next step."
    )
    t0 = time.time()
    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=CHAT_MODEL, max_tokens=250,
    )
    latency = time.time() - t0
    reply = resp.choices[0].message.content
    total_tokens = getattr(resp.usage, "total_tokens", len(reply.split()))
    log_metric("response_generation", CHAT_MODEL, latency,
               tokens=total_tokens)
    return reply.strip()


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------
def run_pipeline(audio, typed_text):
    # 1. ASR (if audio given) else use typed text
    transcript = speech_to_text(audio) if audio else (typed_text or "").strip()
    if not transcript:
        return ("Please record audio or type a complaint.",) + ("",) * 6 + (None,)

    # 3. Sentiment
    sent_display, sent_label, _ = analyze_sentiment(transcript)

    # 4. Intent: zero-shot vs fine-tuned
    zs_label, zs_conf = classify_intent_zeroshot(transcript)
    ft_label, ft_conf = classify_intent_finetuned(transcript)
    intent_display = (f"Zero-shot: {zs_label} ({zs_conf:.2%})\n"
                      f"Fine-tuned: {ft_label} ({ft_conf:.2%})")

    # 5. NER
    entities = extract_entities(transcript)

    # 6. Summarization
    summary = summarize(transcript)

    # 7. Response generation
    reply = generate_response(transcript, sent_label, zs_label)

    # 2. TTS of the reply
    reply_audio = text_to_speech(reply)

    return (transcript, sent_display, intent_display, entities,
            summary, reply, reply_audio)


def record_feedback(feedback):
    log_metric("user_feedback", "human", 0.0,
               confidence=1.0 if feedback == "👍 Helpful" else 0.0,
               extra=feedback)
    return f"Feedback recorded: {feedback}"


# ----------------------------------------------------------------------------
# LLMOps Dashboard
# ----------------------------------------------------------------------------
def load_dashboard():
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame(), "No metrics logged yet — run the pipeline first."
    df = pd.read_csv(METRICS_FILE)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["tokens"] = pd.to_numeric(df["tokens"], errors="coerce")
    df["est_cost_usd"] = pd.to_numeric(df["est_cost_usd"], errors="coerce")

    summary = df.groupby("subtask").agg(
        calls=("subtask", "count"),
        avg_latency_sec=("latency_sec", "mean"),
        avg_confidence=("confidence", "mean"),
        total_tokens=("tokens", "sum"),
        total_cost_usd=("est_cost_usd", "sum"),
    ).round(4).reset_index()

    fb = df[df.subtask == "user_feedback"]
    sat = f"{fb.confidence.mean():.0%}" if len(fb) else "n/a"
    text = (
        f"**Total API calls:** {len(df)}   |   "
        f"**Avg latency:** {df.latency_sec.mean():.2f}s   |   "
        f"**Total tokens:** {int(df.tokens.sum())}   |   "
        f"**Est. cost:** ${df.est_cost_usd.sum():.4f}   |   "
        f"**User satisfaction:** {sat}"
    )
    return summary, text


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
with gr.Blocks(title="VoiceDesk AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎧 VoiceDesk AI — Customer Support Assistant\n"
        "**Domain:** Customer Service  |  **Categories:** NLP + Speech Recognition  |  "
        "**7 AI sub-tasks via Hugging Face APIs**"
    )

    with gr.Tab("📞 Support Pipeline"):
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone", "upload"],
                                    type="filepath",
                                    label="🎙️ Record / upload voice complaint")
                text_in = gr.Textbox(label="…or type the complaint",
                                     placeholder="My order #4521 arrived broken and I want a refund!")
                run_btn = gr.Button("▶ Run Full AI Pipeline", variant="primary")
            with gr.Column():
                transcript_out = gr.Textbox(label="1️⃣ Transcript (Whisper ASR)")
                sentiment_out = gr.Textbox(label="3️⃣ Sentiment")
                intent_out = gr.Textbox(label="4️⃣ Intent — zero-shot vs fine-tuned", lines=2)
                ner_out = gr.Textbox(label="5️⃣ Named Entities")
                summary_out = gr.Textbox(label="6️⃣ Ticket Summary (for human agent)")
        reply_out = gr.Textbox(label="7️⃣ AI-generated Reply", lines=4)
        reply_audio_out = gr.Audio(label="2️⃣ Spoken Reply (TTS)")
        with gr.Row():
            fb = gr.Radio(["👍 Helpful", "👎 Not helpful"], label="Rate this reply (LLMOps feedback metric)")
            fb_msg = gr.Textbox(label="", interactive=False)

        run_btn.click(run_pipeline, [audio_in, text_in],
                      [transcript_out, sentiment_out, intent_out,
                       ner_out, summary_out, reply_out, reply_audio_out])
        fb.change(record_feedback, fb, fb_msg)

    with gr.Tab("📊 LLMOps Dashboard"):
        refresh = gr.Button("🔄 Refresh metrics")
        kpi_md = gr.Markdown()
        table = gr.Dataframe(label="Per sub-task metrics (from metrics_log.csv)")
        refresh.click(load_dashboard, None, [table, kpi_md])

    gr.Markdown("Fine-tuned intent model: DistilBERT trained on the Bitext "
                "Customer Support dataset — see `finetune_intent.py`.")

if __name__ == "__main__":
    demo.launch()
