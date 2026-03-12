"""Tests for the Hardware Recommender Service."""

import pytest
from app.services.hardware_service import recommend_for_hardware, get_hardware_catalog

def test_hardware_catalog_contains_expected_targets():
    catalog = get_hardware_catalog()
    assert len(catalog) >= 6
    ids = [p.id for p in catalog]
    assert "macbook_mseries_8gb" in ids
    assert "enthusiast_gpu_24gb" in ids
    assert "datacenter_gpu_80gb" in ids

def test_recommend_for_low_memory_hardware():
    rec = recommend_for_hardware("raspberry_pi_8gb", task_type="causal_lm")
    assert rec.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert rec.compression_bits == 4
    assert rec.lora_rank == 8
    assert rec.training_batch_size == 1

def test_recommend_for_medium_memory_hardware():
    rec = recommend_for_hardware("consumer_gpu_8gb", task_type="causal_lm")
    assert rec.base_model == "microsoft/phi-2"
    assert rec.compression_bits == 4
    
    rec_class = recommend_for_hardware("consumer_gpu_8gb", task_type="classification")
    assert rec_class.base_model == "Qwen/Qwen2.5-3B-Instruct"
    assert rec_class.compression_bits == 4

def test_recommend_for_high_memory_hardware():
    rec = recommend_for_hardware("enthusiast_gpu_24gb", task_type="causal_lm")
    assert rec.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert rec.compression_bits == 8
    assert rec.lora_rank == 16
    assert rec.training_batch_size == 4

def test_recommend_for_datacenter_hardware():
    rec = recommend_for_hardware("datacenter_gpu_80gb", task_type="causal_lm")
    assert rec.base_model == "meta-llama/Llama-3.1-70B-Instruct"
    assert rec.compression_bits == 4
    assert rec.lora_rank == 32
    assert rec.training_batch_size == 8

def test_recommend_invalid_hardware():
    with pytest.raises(ValueError):
        recommend_for_hardware("invalid_id")
