val prompt = if (args.isNotEmpty()) args.joinToString(" ") else "Hello from Android"
val seed = prompt.sumOf { it.code } % 541
println("Echo: $prompt tok$seed")
