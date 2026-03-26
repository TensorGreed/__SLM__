package com.example.slmreference

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var runtime: SLMRuntime

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        runtime = SLMRuntime(assets)

        val promptInput = findViewById<EditText>(R.id.promptInput)
        val outputText = findViewById<TextView>(R.id.outputText)
        val generateButton = findViewById<Button>(R.id.generateButton)

        generateButton.setOnClickListener {
            val prompt = promptInput.text?.toString().orEmpty()
            outputText.text = runtime.generate(prompt = prompt, maxTokens = 32)
        }
    }
}
