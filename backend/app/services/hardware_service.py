"""Hardware recommender service mapping target devices to model and compression profiles."""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    id: str
    name: str
    description: str
    icon: str  # e.g., 'laptop', 'gpu', 'server'
    max_vram_gb: float

@dataclass
class RecommendationResult:
    base_model: str
    compression_bits: int
    lora_rank: int
    training_batch_size: int
    notes: list[str]

HARDWARE_CATALOG = [
    HardwareProfile(
        id="macbook_mseries_8gb",
        name="MacBook (8GB Unified)",
        description="M-series Mac with 8GB unified memory",
        icon="laptop",
        max_vram_gb=6.0, # leaves OS memory
    ),
    HardwareProfile(
        id="macbook_mseries_16gb",
        name="MacBook (16GB+ Unified)",
        description="M-series Mac with 16GB or more unified memory",
        icon="laptop",
        max_vram_gb=12.0,
    ),
    HardwareProfile(
        id="consumer_gpu_8gb",
        name="Consumer GPU (8GB VRAM)",
        description="e.g., RTX 3060, RTX 4060",
        icon="gpu",
        max_vram_gb=8.0,
    ),
    HardwareProfile(
        id="enthusiast_gpu_24gb",
        name="Enthusiast GPU (24GB VRAM)",
        description="e.g., RTX 3090, RTX 4090",
        icon="gpu",
        max_vram_gb=24.0,
    ),
    HardwareProfile(
        id="datacenter_gpu_80gb",
        name="Datacenter GPU (80GB VRAM)",
        description="e.g., A100, H100",
        icon="server",
        max_vram_gb=80.0,
    ),
    HardwareProfile(
        id="raspberry_pi_8gb",
        name="Raspberry Pi (8GB)",
        description="SBC with 8GB RAM",
        icon="cpu",
        max_vram_gb=4.0, # Very constrained
    )
]

def get_hardware_catalog() -> list[HardwareProfile]:
    """Return the list of predefined hardware deployment targets."""
    return HARDWARE_CATALOG

def recommend_for_hardware(hardware_id: str, task_type: str = "causal_lm") -> RecommendationResult:
    """Recommend model and compression settings based on hardware ID and task type."""
    profile = next((p for p in HARDWARE_CATALOG if p.id == hardware_id), None)
    if not profile:
        raise ValueError(f"Unknown hardware profile ID: {hardware_id}")

    notes = []
    
    # 1. Very small targets (Raspberry Pi, old laptops)
    if profile.max_vram_gb <= 4.0:
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        compression_bits = 4
        lora_rank = 8
        training_batch_size = 1
        notes.append("Target is highly memory constrained. Recommending a sub-1B parameter model.")
        notes.append("4-bit quantization is essential for this target.")
        
    # 2. Medium constraint (8GB Macs, 8GB GPUs)
    elif profile.max_vram_gb <= 8.0:
        base_model = "microsoft/phi-2" if task_type == "causal_lm" else "Qwen/Qwen2.5-3B-Instruct"
        compression_bits = 4
        lora_rank = 8
        training_batch_size = 2
        notes.append("Recommending ~3B parameter model optimized for 8GB environments.")
        notes.append("4-bit quantization recommended to leave room for context window.")

    # 3. High capacity (16GB Macs, 24GB GPUs)
    elif profile.max_vram_gb <= 24.0:
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        compression_bits = 8
        lora_rank = 16
        training_batch_size = 4
        notes.append("Hardware can comfortably fit 8B class models.")
        notes.append("8-bit quantization is a good balance of quality and performance.")

    # 4. Datacenter (80GB+ GPUs)
    else:
        base_model = "meta-llama/Llama-3.1-70B-Instruct"
        compression_bits = 4 # Even 70B needs quantization to fit comfortably with context on 80GB
        lora_rank = 32
        training_batch_size = 8
        notes.append("Hardware has massive VRAM capacity.")
        notes.append("Recommending 70B class model with 4-bit quantization, or you could run an 8B model completely unquantized.")

    if task_type == "classification":
        notes.append("For classification tasks, encoder models (like BERT) might be an even more efficient alternative.")

    return RecommendationResult(
        base_model=base_model,
        compression_bits=compression_bits,
        lora_rank=lora_rank,
        training_batch_size=training_batch_size,
        notes=notes
    )
