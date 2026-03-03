"""
SLM Platform Product Guide PDF Generator
Generates a comprehensive, professional multi-page PDF with embedded screenshots.
"""
from fpdf import FPDF
import os

BRAIN = r"C:\Users\Administrator\.gemini\antigravity\brain\8f335972-bf27-4b45-beb5-aac29c3c8011"

SCREENSHOTS = {
    "cover": os.path.join(BRAIN, "slm_product_guide_cover_1772563179261.png"),
    "data_local": os.path.join(BRAIN, "data_ingestion_local_files_1772564593846.png"),
    "data_hf": os.path.join(BRAIN, "data_ingestion_huggingface_tab_1772564603215.png"),
    "cleaning": os.path.join(BRAIN, "data_cleaning_panel_1772564612922.png"),
    "goldset": os.path.join(BRAIN, "gold_set_panel_final_1772564622651.png"),
    "synthetic": os.path.join(BRAIN, "synthetic_data_panel_1772564632279.png"),
    "training": os.path.join(BRAIN, "training_panel_final_1772564663784.png"),
    "training_live": os.path.join(BRAIN, "training_dashboard_live_final_1772562142363.png"),
    "evaluation": os.path.join(BRAIN, "evaluation_panel_final_1772564672954.png"),
    "llm_judge": os.path.join(BRAIN, "llm_benchmarking_results_1772562939845.png"),
    "compression": os.path.join(BRAIN, "compression_panel_final_1772564682445.png"),
    "export": os.path.join(BRAIN, "export_panel_final_1772564692913.png"),
}


class ProductGuide(FPDF):
    def header(self):
        if self.page_no() > 2:
            self.set_font("helvetica", "I", 8)
            self.set_text_color(120, 120, 140)
            self.cell(0, 8, "SLM Platform  |  Product Guide v1.0", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(80, 80, 120)
            self.line(10, 12, 200, 12)
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 140)
        if self.page_no() > 1:
            self.cell(0, 10, f"Page {self.page_no() - 1}", align="C")

    def chapter_title(self, num, title):
        self.set_font("helvetica", "B", 20)
        self.set_text_color(90, 60, 200)
        self.cell(0, 14, f"Chapter {num}", align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_font("helvetica", "B", 16)
        self.set_text_color(30, 30, 50)
        self.cell(0, 10, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(90, 60, 200)
        self.set_line_width(0.8)
        self.line(10, self.get_y() + 2, 80, self.get_y() + 2)
        self.set_line_width(0.2)
        self.ln(8)

    def section_title(self, title):
        self.set_font("helvetica", "B", 13)
        self.set_text_color(50, 50, 80)
        self.cell(0, 10, title, align="L", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(40, 40, 60)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def step(self, num, title, description):
        self.set_font("helvetica", "B", 11)
        self.set_text_color(90, 60, 200)
        self.cell(8, 7, str(num) + ".", new_x="RIGHT", new_y="TOP")
        self.set_text_color(30, 30, 50)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        if description:
            self.set_x(18)
            self.set_font("helvetica", "", 10)
            self.set_text_color(60, 60, 80)
            self.multi_cell(172, 5.5, description)
        self.ln(2)

    def add_screenshot(self, key, caption=""):
        path = SCREENSHOTS.get(key)
        if path and os.path.exists(path):
            img_w = 180
            self.image(path, x=15, w=img_w)
            if caption:
                self.set_font("helvetica", "I", 8)
                self.set_text_color(100, 100, 120)
                self.cell(0, 6, caption, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(6)

    def tip_box(self, text):
        self.set_fill_color(235, 230, 255)
        self.set_draw_color(90, 60, 200)
        self.set_font("helvetica", "B", 9)
        self.set_text_color(90, 60, 200)
        x = self.get_x()
        y = self.get_y()
        self.rect(10, y, 190, 18, style="DF")
        self.set_xy(14, y + 2)
        self.cell(0, 6, "TIP", new_x="LMARGIN", new_y="NEXT")
        self.set_x(14)
        self.set_font("helvetica", "", 9)
        self.set_text_color(60, 50, 100)
        self.multi_cell(182, 5, text)
        self.ln(6)


def build_guide():
    pdf = ProductGuide()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ────────────────────────────── COVER ──────────────────────────────
    pdf.add_page()
    if os.path.exists(SCREENSHOTS["cover"]):
        pdf.image(SCREENSHOTS["cover"], x=0, y=0, w=210, h=297)

    # ────────────────────────────── TITLE PAGE ──────────────────────────────
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("helvetica", "B", 32)
    pdf.set_text_color(90, 60, 200)
    pdf.cell(0, 18, "SLM Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 16)
    pdf.set_text_color(80, 80, 100)
    pdf.cell(0, 10, "Product Guide for ML Engineers", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(90, 60, 200)
    pdf.set_line_width(1)
    pdf.line(70, pdf.get_y(), 140, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(10)
    pdf.set_font("helvetica", "", 11)
    pdf.set_text_color(100, 100, 120)
    pdf.cell(0, 8, "Version 1.0  |  2026 Edition", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Optimized for NVIDIA DGX Spark", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(140, 140, 160)
    pdf.cell(0, 6, "Confidential & Proprietary", align="C", new_x="LMARGIN", new_y="NEXT")

    # ────────────────────────────── TABLE OF CONTENTS ──────────────────────────────
    pdf.add_page()
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(30, 30, 50)
    pdf.cell(0, 14, "Table of Contents", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    toc = [
        ("1", "Getting Started", "Project creation and platform overview"),
        ("2", "Data Ingestion", "File uploads, HuggingFace, Kaggle, and URL connectors"),
        ("3", "Data Cleaning & PII Redaction", "Deduplication, quality filtering, and privacy"),
        ("4", "Gold Dataset Builder", "Curating ground-truth evaluation sets"),
        ("5", "Synthetic Data Generation", "AI-powered Q&A pair generation"),
        ("6", "Training & Fine-Tuning", "LoRA configuration, optimizers, and live telemetry"),
        ("7", "Evaluation & LLM-as-a-Judge", "Benchmarking, radar analytics, side-by-side arena"),
        ("8", "Compression & Quantization", "4-bit, 8-bit weight optimization"),
        ("9", "Export & Deployment", "GGUF, ONNX, and HuggingFace Hub publishing"),
        ("A", "Architecture Reference", "API endpoints and system design"),
    ]
    for num, title, desc in toc:
        pdf.set_font("helvetica", "B", 11)
        pdf.set_text_color(90, 60, 200)
        pdf.cell(12, 8, num, new_x="RIGHT", new_y="TOP")
        pdf.set_text_color(30, 30, 50)
        pdf.cell(70, 8, title, new_x="RIGHT", new_y="TOP")
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(0, 8, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ──────────────── CH 1: GETTING STARTED ────────────────
    pdf.add_page()
    pdf.chapter_title(1, "Getting Started")

    pdf.body_text(
        "The SLM Platform is a production-grade orchestration engine for fine-tuning, evaluating, and deploying "
        "Small Language Models. It is designed for ML engineers who need granular control over every stage of the "
        "model lifecycle, from raw data ingestion through to quantized GGUF export."
    )
    pdf.body_text(
        "The platform runs as a decoupled frontend + backend system. The React dashboard communicates with a "
        "FastAPI backend that orchestrates all compute-intensive tasks via PyTorch and HuggingFace Transformers."
    )

    pdf.section_title("System Requirements")
    pdf.step(1, "Hardware", "NVIDIA GPU with 16GB+ VRAM (DGX Spark recommended). 32GB+ system RAM.")
    pdf.step(2, "Software", "Python 3.11+, Node.js 20+, CUDA 12.x toolkit.")
    pdf.step(3, "Dependencies", "PyTorch 2.x, HuggingFace Transformers, PEFT, bitsandbytes, vLLM (optional).")

    pdf.section_title("Launching the Platform")
    pdf.step(1, "Start the Backend", "cd backend && python -m uvicorn app.main:app --reload --port 8000")
    pdf.step(2, "Start the Frontend", "cd frontend && npm run dev")
    pdf.step(3, "Open the Dashboard", "Navigate to http://localhost:5173 in your browser.")
    pdf.step(4, "Create a Project", "Click 'Create New Project', name it (e.g. 'Customer Support SLM'), and enter the project.")

    pdf.tip_box("The sidebar shows all 10 pipeline stages. Work through them left-to-right for a complete fine-tuning workflow.")

    # ──────────────── CH 2: DATA INGESTION ────────────────
    pdf.add_page()
    pdf.chapter_title(2, "Data Ingestion")

    pdf.body_text(
        "The Data Ingestion panel is the entry point for all training data. It supports four source types, "
        "accessible via tabs at the top of the panel."
    )

    pdf.section_title("Local File Upload")
    pdf.step(1, "Select 'Local Files' Tab", "This is the default view. You will see a drag-and-drop upload zone.")
    pdf.step(2, "Upload Documents", "Drag files into the zone or click 'Choose Files'. Supported formats: PDF, DOCX, TXT, Markdown, CSV, JSON, JSONL.")
    pdf.step(3, "Process Documents", "After upload, each file appears in the Ingested Documents table with status 'pending'. Click the gear icon to parse and extract text.")

    pdf.add_screenshot("data_local", "Figure 2.1 -- Data Ingestion panel with Local Files tab and source connectors")

    pdf.add_page()
    pdf.section_title("HuggingFace Hub Connector")
    pdf.step(1, "Click the 'HuggingFace' Tab", "The form switches to show a Dataset Identifier field and Split selector.")
    pdf.step(2, "Enter Dataset ID", "Type the HuggingFace dataset path, e.g. 'tatsu-lab/alpaca', 'squad', or 'openai/gsm8k'.")
    pdf.step(3, "Select Split", "Choose train, test, or validation from the dropdown.")
    pdf.step(4, "Set Max Samples (Optional)", "Limit the number of rows to import for faster iteration.")
    pdf.step(5, "Click 'Import from HuggingFace'", "The backend downloads and converts the dataset to JSONL format.")

    pdf.add_screenshot("data_hf", "Figure 2.2 -- HuggingFace Hub connector with dataset identifier and split selector")

    pdf.section_title("Kaggle Connector")
    pdf.step(1, "Click the 'Kaggle' Tab", "Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables on the server.")
    pdf.step(2, "Enter Dataset Slug", "Use the format 'username/dataset-name' as shown on Kaggle.")
    pdf.step(3, "Click 'Import from Kaggle'", "The dataset is downloaded and converted to JSONL.")

    pdf.section_title("URL / S3 / GCS Connector")
    pdf.step(1, "Click the 'URL / S3 / GCS' Tab", "For importing data from any direct HTTP link or cloud storage URI.")
    pdf.step(2, "Paste the URL", "Supports direct links to .csv, .json, or .jsonl files, as well as S3 and GCS URIs.")
    pdf.step(3, "Click 'Import from URL'", "The file is fetched and stored in the project's data directory.")

    pdf.tip_box("All imported datasets appear in the same Ingested Documents table. The 'Source' column shows where each file came from (e.g. huggingface:tatsu-lab/alpaca).")

    # ──────────────── CH 3: DATA CLEANING ────────────────
    pdf.add_page()
    pdf.chapter_title(3, "Data Cleaning & PII Redaction")

    pdf.body_text(
        "Raw data often contains duplicates, noise, and sensitive personally identifiable information (PII). "
        "The Cleaning module provides automated tools to sanitize your dataset before training."
    )

    pdf.section_title("Step-by-Step")
    pdf.step(1, "Navigate to 'Cleaning'", "Click the Cleaning tab in the left sidebar.")
    pdf.step(2, "Select a Dataset", "Choose the raw dataset you want to clean from the dropdown.")
    pdf.step(3, "Configure Chunking", "Set the chunk size (characters) and overlap. Smaller chunks work better for Q&A; larger chunks for summarization.")
    pdf.step(4, "Enable Deduplication", "Check 'Exact Dedup' for hash-based removal, or 'Semantic Dedup' for embedding-based near-duplicate detection.")
    pdf.step(5, "Enable PII Redaction", "Automatically masks emails, phone numbers, SSNs, and other PII patterns.")
    pdf.step(6, "Run Cleaning Job", "Click the action button. Progress is shown in real-time.")

    pdf.add_screenshot("cleaning", "Figure 3.1 -- Data Cleaning panel with chunking strategy and PII redaction options")

    pdf.tip_box("PII redaction uses regex patterns and NER models to detect sensitive data. Always review a sample of cleaned records before training.")

    # ──────────────── CH 4: GOLD SET ────────────────
    pdf.add_page()
    pdf.chapter_title(4, "Gold Dataset Builder")

    pdf.body_text(
        "Gold datasets are hand-curated, high-quality Q&A pairs that serve as the ground truth for evaluation. "
        "A strong gold set is critical for measuring model quality after fine-tuning."
    )

    pdf.section_title("Step-by-Step")
    pdf.step(1, "Navigate to 'Gold Set'", "Click the Gold Set tab in the sidebar.")
    pdf.step(2, "Add Q&A Pairs Manually", "Enter a question in the 'Question' field and the expected answer in the 'Answer' field.")
    pdf.step(3, "Set Difficulty Level", "Tag each pair as Easy, Medium, or Hard to enable stratified evaluation.")
    pdf.step(4, "Import from Existing Data", "You can also auto-generate gold set candidates from your cleaned dataset.")
    pdf.step(5, "Review and Approve", "Each pair has an approve/reject button. Only approved pairs are used for evaluation.")
    
    pdf.ln(5)
    pdf.section_title("Real-World Example")
    pdf.body_text("If building a customer support SLM, your Gold Set should contain exact expert answers:")
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Q: What is the refund policy for annual subscriptions cancelled after 30 days?", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, "A: Annual subscriptions cancelled after 30 days are non-refundable, but the account remains active until the end of the billing cycle. (See Terms section 4.2)")
    pdf.ln(3)

    pdf.add_screenshot("goldset", "Figure 4.1 -- Gold Dataset Builder interface")

    # ──────────────── CH 5: SYNTHETIC GENERATION ────────────────
    pdf.add_page()
    pdf.chapter_title(5, "Synthetic Data Generation")

    pdf.body_text(
        "When manual data creation is too slow, use the Synthetic Generation module. A large Teacher model "
        "(e.g. Llama-3-70B) reads your source documents and automatically generates diverse, high-quality Q&A pairs."
    )

    pdf.section_title("Step-by-Step")
    pdf.step(1, "Navigate to 'Synthetic'", "Click the Synthetic tab in the sidebar.")
    pdf.step(2, "Select Teacher Model", "Choose a strong model from the dropdown (e.g. meta-llama/Meta-Llama-3-70B-Instruct).")
    pdf.step(3, "Configure Generation", "Set the number of pairs to generate, temperature, and context window size.")
    pdf.step(4, "Run Generation", "Click the generate button. The Teacher model processes your documents and creates Q&A pairs.")
    pdf.step(5, "Review Output", "Generated pairs appear in a table. Approve, edit, or reject each one before adding to your training set.")

    pdf.ln(5)
    pdf.section_title("Local Deployment via Ollama")
    pdf.body_text("The platform supports local inference via Ollama. If you have a local model (e.g. Llama-3 or Mixtral) running on localhost:11434, you can use it as the Teacher model to ensure no data leaves your network.")
    
    pdf.ln(3)
    pdf.section_title("How Generation Works")
    pdf.body_text("1. Source Text from your raw documents:")
    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(0, 5, "\"The HTTP 429 Too Many Requests response status code indicates the user has sent too many requests in a given amount of time ('rate limiting').\"")
    pdf.set_font("helvetica", "", 10)
    pdf.ln(2)
    pdf.body_text("2. Teacher Output (Generated Training Pair):")
    pdf.set_font("helvetica", "I", 10)
    pdf.multi_cell(0, 5, "Q: What does an HTTP 429 status code mean?\nA: It means Too Many Requests, indicating the user has hit a rate limit.")
    pdf.set_font("helvetica", "", 10)
    pdf.ln(3)

    pdf.add_screenshot("synthetic", "Figure 5.1 -- Synthetic Data Generation panel with Teacher model configuration")

    pdf.tip_box("Use a high temperature (0.8-1.0) for diverse question styles. Use low temperature (0.2-0.4) for factual, precise pairs.")

    # ──────────────── CH 6: TRAINING ────────────────
    pdf.add_page()
    pdf.chapter_title(6, "Training & Fine-Tuning")

    pdf.body_text(
        "The Training Dashboard provides full control over Parameter-Efficient Fine-Tuning (PEFT) using LoRA/QLoRA. "
        "It includes real-time WebSocket telemetry for monitoring loss, learning rate, and GPU utilization."
    )

    pdf.section_title("Configuring a Training Run")
    pdf.step(1, "Navigate to 'Training'", "Click the Training tab in the sidebar.")
    pdf.step(2, "Create New Experiment", "Click 'New Experiment'. Name it descriptively (e.g. 'v1-customer-support-lora').")
    pdf.step(3, "Set Base Model", "Enter the HuggingFace model ID (e.g. meta-llama/Meta-Llama-3-8B).")
    pdf.step(4, "Configure LoRA Parameters", "")
    pdf.body_text(
        "   - Rank (r): Controls adapter capacity. Higher = more expressive but more VRAM. Start with 16.\n"
        "   - Alpha: Scaling factor. Set to 2x rank (e.g. Alpha=32 for Rank=16).\n"
        "   - Target Modules: q_proj, v_proj (default). Add k_proj, o_proj for deeper adaptation.\n"
        "   - LoRA Dropout: 0.05-0.1 for regularization."
    )
    pdf.step(5, "Select Optimizer", "Paged AdamW 8-bit is recommended for VRAM efficiency on consumer GPUs. Standard AdamW for DGX-class hardware.")
    pdf.step(6, "Set Learning Rate Schedule", "Cosine with warmup is the default. Linear decay works for shorter runs.")
    pdf.step(7, "Enable Flash Attention 2", "Check this box if your GPU supports it (Ampere+). Reduces memory usage by 2-4x.")
    pdf.step(8, "Launch Training", "Click 'Start Training'. The live dashboard will appear.")

    pdf.add_screenshot("training", "Figure 6.1 -- Training Dashboard with experiment configuration")

    pdf.add_page()
    pdf.section_title("Live Telemetry Dashboard")
    pdf.body_text(
        "Once training starts, the dashboard shows real-time charts powered by WebSockets. You will see:"
    )
    pdf.step(1, "Training Loss Curve", "Shows the loss value at each training step. A healthy curve drops steeply then flattens.")
    pdf.step(2, "Learning Rate Schedule", "Visualizes the warmup and decay phases of your LR scheduler.")
    pdf.step(3, "GPU Utilization", "Shows CUDA memory usage and compute utilization percentage.")
    pdf.step(4, "Step Counter", "Tracks total steps completed vs. target, with estimated time remaining.")

    pdf.add_screenshot("training_live", "Figure 6.2 -- Live training telemetry with loss curves and GPU metrics")

    pdf.tip_box("If loss plateaus early, try increasing the learning rate or LoRA rank. If loss spikes, reduce the learning rate by 50%.")

    # ──────────────── CH 7: EVALUATION ────────────────
    pdf.add_page()
    pdf.chapter_title(7, "Evaluation & LLM-as-a-Judge")

    pdf.body_text(
        "The Evaluation module provides both traditional metric-based evaluation and advanced LLM-as-a-Judge "
        "benchmarking. A strong Judge model (e.g. Llama-3-70B) scores your SLM's outputs and provides technical rationale."
    )

    pdf.section_title("Running Standard Evaluation")
    pdf.step(1, "Navigate to 'Evaluation'", "Click the Evaluation tab in the sidebar.")
    pdf.step(2, "Select an Experiment", "Choose a completed training experiment from the list.")
    pdf.step(3, "Run Evaluation", "Click 'Run Evaluation'. The system tests against your gold set and computes metrics (BLEU, ROUGE, exact match).")

    pdf.add_screenshot("evaluation", "Figure 7.1 -- Evaluation panel with experiment selection")

    pdf.add_page()
    pdf.section_title("LLM-as-a-Judge Benchmarking")
    pdf.step(1, "Click '+ Run LLM Benchmark'", "Opens the benchmark configuration form.")
    pdf.step(2, "Select Judge Model", "Choose a strong evaluator (default: meta-llama/Meta-Llama-3-70B-Instruct).")
    pdf.step(3, "Select Benchmark Suite", "Choose from MMLU-Subset, HumanEval, GSM8K, or custom datasets.")
    pdf.step(4, "Click 'Run Benchmark'", "The Judge model evaluates each prediction on a 1-5 scale.")
    pdf.step(5, "Review Radar Chart", "A skill radar visualizes pass rates across different capability domains (Factuality, Logic, Safety, Tone, Helpfulness).")
    pdf.step(6, "Inspect Side-by-Side Arena", "Compare Base Model vs. Trained SLM outputs with the Judge's score and rationale for each prompt.")

    pdf.add_screenshot("llm_judge", "Figure 7.2 -- LLM-as-a-Judge results: Radar analytics and side-by-side comparison arena")

    pdf.tip_box("The side-by-side arena is particularly useful for identifying specific failure modes. Look for prompts where your SLM scores below 3.")

    # ──────────────── CH 8: COMPRESSION ────────────────
    pdf.add_page()
    pdf.chapter_title(8, "Compression & Quantization")

    pdf.body_text(
        "After training, compress the model for faster inference and smaller deployment footprint. "
        "The platform supports multiple quantization strategies."
    )

    pdf.section_title("Step-by-Step")
    pdf.step(1, "Navigate to 'Compression'", "Click the Compression tab in the sidebar.")
    pdf.step(2, "Select Quantization Level", "Choose 4-bit (smallest, slight quality loss) or 8-bit (balanced).")
    pdf.step(3, "Select Method", "GPTQ for GPU inference, AWQ for optimized deployment, or BitsAndBytes for on-the-fly quantization.")
    pdf.step(4, "Merge LoRA Weights", "Check 'Merge before quantization' to bake the LoRA adapters into the base weights.")
    pdf.step(5, "Run Compression", "Click compress. The output will be a standalone model ready for export.")

    pdf.add_screenshot("compression", "Figure 8.1 -- Compression panel with quantization options")

    pdf.tip_box("4-bit GPTQ models run 3-4x faster than FP16 on consumer GPUs with minimal quality loss. Always benchmark after quantization.")

    # ──────────────── CH 9: EXPORT ────────────────
    pdf.add_page()
    pdf.chapter_title(9, "Export & Deployment")

    pdf.body_text(
        "The final stage packages your fine-tuned, compressed model into a deployment-ready format."
    )

    pdf.section_title("Step-by-Step")
    pdf.step(1, "Navigate to 'Export'", "Click the Export tab in the sidebar.")
    pdf.step(2, "Select Export Format", "")
    pdf.body_text(
        "   - GGUF: For local inference with llama.cpp or Ollama.\n"
        "   - ONNX: For cross-platform deployment with ONNX Runtime.\n"
        "   - Safetensors: Standard HuggingFace format for cloud serving.\n"
        "   - Docker Container: Auto-generates a Dockerfile with inference server."
    )
    pdf.step(3, "Configure Metadata", "Set model name, description, and license for the model card.")
    pdf.step(4, "Push to HuggingFace Hub (Optional)", "Enter your HF token and repository name to publish directly.")
    pdf.step(5, "Download Locally", "Click 'Export' to download the packaged model to your local machine.")

    pdf.add_screenshot("export", "Figure 9.1 -- Export panel with format selection and Hub publishing")

    # ──────────────── APPENDIX A: ARCHITECTURE ────────────────
    pdf.add_page()
    pdf.chapter_title("A", "Architecture Reference")

    pdf.section_title("System Architecture")
    pdf.body_text(
        "Frontend: React 18 + Vite + TypeScript + Zustand (state) + Recharts (visualization)\n"
        "Backend: Python 3.11 + FastAPI + SQLAlchemy (async) + PyTorch + HuggingFace\n"
        "Database: SQLite (development) / PostgreSQL (production)\n"
        "Compute: NVIDIA CUDA with Flash Attention 2, bitsandbytes, PEFT"
    )

    pdf.section_title("Key API Endpoints")
    endpoints = [
        ("POST", "/projects/{id}/ingestion/upload-batch", "Upload local files"),
        ("POST", "/projects/{id}/ingestion/import-remote", "Import from HuggingFace/Kaggle/URL"),
        ("POST", "/projects/{id}/cleaning/run", "Run data cleaning pipeline"),
        ("POST", "/projects/{id}/training/start", "Launch training job"),
        ("GET", "/experiments/{id}/metrics/ws", "WebSocket for live metrics"),
        ("POST", "/projects/{id}/evaluation/llm-judge", "Run LLM-as-a-Judge benchmark"),
        ("POST", "/projects/{id}/export/run", "Export model to specified format"),
    ]
    for method, path, desc in endpoints:
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(90, 60, 200)
        pdf.cell(14, 6, method, new_x="RIGHT", new_y="TOP")
        pdf.set_font("courier", "", 8)
        pdf.set_text_color(40, 40, 60)
        pdf.cell(90, 6, path, new_x="RIGHT", new_y="TOP")
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(0, 6, desc, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.section_title("Environment Variables")
    envs = [
        ("DATA_DIR", "Path to data storage directory"),
        ("DATABASE_URL", "SQLAlchemy connection string"),
        ("KAGGLE_USERNAME", "Kaggle API username (for Kaggle connector)"),
        ("KAGGLE_KEY", "Kaggle API key"),
        ("HF_TOKEN", "HuggingFace access token (for gated models and Hub push)"),
        ("CUDA_VISIBLE_DEVICES", "Select which GPUs to use (e.g. 0,1)"),
    ]
    for name, desc in envs:
        pdf.set_font("courier", "B", 9)
        pdf.set_text_color(40, 40, 60)
        pdf.cell(50, 6, name, new_x="RIGHT", new_y="TOP")
        pdf.set_font("helvetica", "", 9)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(0, 6, desc, new_x="LMARGIN", new_y="NEXT")

    # ──────────────── SAVE ────────────────
    out = os.path.join(BRAIN, "slm_product_guide.pdf")
    pdf.output(out)
    print(f"Product Guide generated: {out}")
    print(f"Total pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_guide()
