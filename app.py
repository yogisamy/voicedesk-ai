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

# LLMOps: MLflow experiment tracking (optional — app still works without it)
# Pin the store to <project>/mlflow.db so logging and `mlflow ui
# --backend-store-uri sqlite:///mlflow.db` always see the same data.
try:
    import mlflow
    _DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{_DB}")
    mlflow.set_experiment("voicedesk-ai")
    MLFLOW_ON = True
except Exception:
    MLFLOW_ON = False

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
    if MLFLOW_ON and mlflow.active_run():
        mlflow.log_metric(f"{subtask}_latency_sec", round(latency, 3))
        if confidence is not None:
            mlflow.log_metric(f"{subtask}_confidence", round(confidence, 4))
        if tokens:
            mlflow.log_metric(f"{subtask}_tokens", tokens)
            mlflow.log_metric(f"{subtask}_est_cost_usd", cost)


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
    scores = {r.label: r.score for r in result}
    log_metric("sentiment", SENTIMENT_MODEL, latency, confidence=top.score)
    return scores, top.label, top.score


# ----------------------------------------------------------------------------
# Sub-task 4a: Intent Classification — zero-shot baseline (NLP)
# ----------------------------------------------------------------------------
def classify_intent_zeroshot(text):
    result, latency = timed("intent_zeroshot", ZEROSHOT_MODEL,
                            client.zero_shot_classification, text,
                            candidate_labels=CANDIDATE_INTENTS,
                            model=ZEROSHOT_MODEL)
    top = result[0]
    scores = {r.label: r.score for r in result}
    log_metric("intent_zeroshot", ZEROSHOT_MODEL, latency, confidence=top.score)
    return scores, top.label, top.score


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
    ents = [{"entity": e.entity_group, "start": e.start, "end": e.end}
            for e in result]
    avg_conf = sum(e.score for e in result) / len(result) if result else 0
    log_metric("ner", NER_MODEL, latency, confidence=avg_conf,
               extra=f"entities={len(ents)}")
    return {"text": text, "entities": ents}


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
    """Generator: yields after every sub-task so the UI fills in live."""
    out = {"transcript": "", "sentiment": None, "intent_zs": None,
           "intent_ft": None, "entities": None, "summary": "",
           "reply": "", "audio": None}

    def snap(status):
        return (out["transcript"], out["sentiment"], out["intent_zs"],
                out["intent_ft"], out["entities"], out["summary"],
                out["reply"], out["audio"], status)

    yield snap("⏳ **Step 1/7** — transcribing audio…" if audio
               else "⏳ **Step 1/7** — reading complaint…")
    if MLFLOW_ON:
        mlflow.start_run(
            run_name=f"pipeline-{datetime.datetime.now():%Y%m%d-%H%M%S}")
        mlflow.log_param("input_mode", "voice" if audio else "text")
    transcript = speech_to_text(audio) if audio else (typed_text or "").strip()
    if not transcript:
        if MLFLOW_ON:
            mlflow.end_run()
        yield snap("⚠️ Please record audio or type a complaint first.")
        return
    out["transcript"] = transcript
    if MLFLOW_ON:
        mlflow.log_param("transcript_words", len(transcript.split()))
    try:
        yield snap("⏳ **Step 2/7** — analysing sentiment…")
        sent_scores, sent_label, _ = analyze_sentiment(transcript)
        out["sentiment"] = sent_scores

        yield snap("⏳ **Step 3/7** — classifying intent (zero-shot + fine-tuned)…")
        zs_scores, zs_label, _ = classify_intent_zeroshot(transcript)
        out["intent_zs"] = zs_scores
        ft_label, ft_conf = classify_intent_finetuned(transcript)
        out["intent_ft"] = {ft_label: ft_conf}

        yield snap("⏳ **Step 4/7** — extracting named entities…")
        out["entities"] = extract_entities(transcript)

        yield snap("⏳ **Step 5/7** — summarising the ticket…")
        out["summary"] = summarize(transcript)

        yield snap("⏳ **Step 6/7** — drafting the reply…")
        out["reply"] = generate_response(transcript, sent_label, zs_label)

        yield snap("⏳ **Step 7/7** — converting reply to speech…")
        out["audio"] = text_to_speech(out["reply"])
    except Exception as e:
        if MLFLOW_ON:
            mlflow.set_tag("status", "failed")
        yield snap(f"❌ **{type(e).__name__}** — {e}")
        return
    finally:
        if MLFLOW_ON:
            mlflow.end_run()

    yield snap("✅ **Done** — rate the reply below to log the feedback metric 👇")


def record_feedback(feedback):
    score = 1.0 if feedback == "👍 Helpful" else 0.0
    log_metric("user_feedback", "human", 0.0, confidence=score, extra=feedback)
    if MLFLOW_ON:
        with mlflow.start_run(run_name="user-feedback"):
            mlflow.log_metric("user_feedback", score)
    return f"Feedback recorded: {feedback}"


# ----------------------------------------------------------------------------
# LLMOps Dashboard
# ----------------------------------------------------------------------------
def load_dashboard():
    empty_lat = pd.DataFrame({"subtask": [], "avg_latency_sec": []})
    empty_conf = pd.DataFrame({"subtask": [], "avg_confidence": []})
    if not os.path.exists(METRICS_FILE):
        return (pd.DataFrame(), "No metrics logged yet — run the pipeline first.",
                empty_lat, empty_conf)
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
    lat_df = summary[["subtask", "avg_latency_sec"]]
    conf_df = summary.dropna(subset=["avg_confidence"])[["subtask", "avg_confidence"]]
    return summary, text, lat_df, conf_df


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------
SAMPLE_COMPLAINTS = [
    "Hi, this is Priya. My order 4521 from Chennai arrived damaged and I want a refund.",
    "I was charged twice by HDFC Bank for my March subscription — please fix the payment.",
    "My BlueDart package never arrived in Mumbai and the tracking page shows nothing.",
    "John from support was great, but I still cannot log in to my account after the reset.",
]

with gr.Blocks(title="VoiceDesk AI") as demo:
    gr.Markdown(
        "# 🎧 VoiceDesk AI — Customer Support Assistant\n"
        "**Domain:** Customer Service  |  **Categories:** NLP + Speech Recognition  |  "
        "**7 AI sub-tasks via Hugging Face APIs**"
    )

    with gr.Tab("📞 Support Pipeline"):
        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(sources=["microphone", "upload"],
                                    type="filepath",
                                    label="🎙️ Record / upload voice complaint")
                text_in = gr.Textbox(label="…or type the complaint", lines=3,
                                     placeholder="My order #4521 arrived broken and I want a refund!")
                gr.Examples(SAMPLE_COMPLAINTS, inputs=text_in,
                            label="💡 One-click sample complaints")
                run_btn = gr.Button("▶ Run Full AI Pipeline",
                                    variant="primary", size="lg")
                status_md = gr.Markdown()
            with gr.Column(scale=2):
                transcript_out = gr.Textbox(label="1️⃣ Transcript (Whisper ASR)")
                with gr.Row():
                    sentiment_out = gr.Label(label="3️⃣ Sentiment (DistilBERT-SST-2)",
                                             num_top_classes=2)
                    intent_ft_out = gr.Label(label="4️⃣b Intent — our fine-tuned DistilBERT")
                intent_zs_out = gr.Label(label="4️⃣a Intent — zero-shot BART-MNLI",
                                         num_top_classes=3)
                ner_out = gr.HighlightedText(label="5️⃣ Named Entities (BERT-NER)",
                                             combine_adjacent=True)
                summary_out = gr.Textbox(label="6️⃣ Ticket Summary (for human agent)")
        with gr.Row():
            with gr.Column(scale=2):
                reply_out = gr.Textbox(label="7️⃣ AI-generated Reply (Qwen2.5-7B)",
                                       lines=4)
            with gr.Column(scale=1):
                reply_audio_out = gr.Audio(label="2️⃣ Spoken Reply (gTTS)",
                                           autoplay=True)
        with gr.Row():
            fb = gr.Radio(["👍 Helpful", "👎 Not helpful"],
                          label="Rate this reply (LLMOps feedback metric)")
            fb_msg = gr.Textbox(label="", interactive=False)

        run_btn.click(run_pipeline, [audio_in, text_in],
                      [transcript_out, sentiment_out, intent_zs_out, intent_ft_out,
                       ner_out, summary_out, reply_out, reply_audio_out, status_md])
        fb.change(record_feedback, fb, fb_msg)

    with gr.Tab("📊 LLMOps Dashboard") as dash_tab:
        refresh = gr.Button("🔄 Refresh metrics", scale=0)
        kpi_md = gr.Markdown()
        with gr.Row():
            lat_plot = gr.BarPlot(x="subtask", y="avg_latency_sec", sort="-y",
                                  title="Avg latency per sub-task (seconds)",
                                  x_title="", y_title="seconds", height=300)
            conf_plot = gr.BarPlot(x="subtask", y="avg_confidence", sort="-y",
                                   title="Avg confidence / quality per sub-task",
                                   x_title="", y_title="score (0–1)", height=300)
        table = gr.Dataframe(label="Per sub-task metrics (from metrics_log.csv)")
        dash_outs = [table, kpi_md, lat_plot, conf_plot]
        refresh.click(load_dashboard, None, dash_outs)
        dash_tab.select(load_dashboard, None, dash_outs)

    gr.Markdown("Fine-tuned intent model: DistilBERT trained on the Bitext "
                "Customer Support dataset — see `finetune_intent.py`.")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
