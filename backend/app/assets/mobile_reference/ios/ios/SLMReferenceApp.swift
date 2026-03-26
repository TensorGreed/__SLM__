import SwiftUI

@main
struct SLMReferenceApp: App {
    @State private var prompt: String = "Hello from iOS"
    @State private var output: String = ""

    private let runtime = SLMRuntime(modelDirectoryURL: Bundle.main.resourceURL?.appendingPathComponent("ModelAssets"))

    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 12) {
                Text("SLM iOS Reference")
                    .font(.headline)
                Text("Model path: ios/ModelAssets/model.mlmodelc")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                TextField("Prompt", text: $prompt)
                    .textFieldStyle(.roundedBorder)

                Button("Generate") {
                    output = runtime.generate(prompt: prompt, maxTokens: 32)
                }
                .buttonStyle(.borderedProminent)

                ScrollView {
                    Text(output)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .font(.body)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(minHeight: 160)
            }
            .padding(20)
        }
    }
}
