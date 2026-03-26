package com.example.slmreference

import android.content.res.AssetManager

class SLMRuntime(private val assets: AssetManager) {
    fun generate(prompt: String, maxTokens: Int = 32): String {
        val base = prompt.trim().ifBlank { "hello" }
        val modelSize = readModelSizeBytes("models/model.bin")
        val seed = (base.sumOf { it.code } + modelSize) % 9973
        val bounded = maxOf(4, minOf(maxTokens, 24))

        val tokens = mutableListOf("Echo:", base)
        repeat(bounded) { index ->
            val value = (seed + (index * 37)) % 541
            tokens += "tok$value"
        }
        return tokens.joinToString(" ")
    }

    private fun readModelSizeBytes(assetPath: String): Int {
        return try {
            assets.openFd(assetPath).length.toInt()
        } catch (_: Exception) {
            0
        }
    }
}
