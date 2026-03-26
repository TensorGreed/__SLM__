import Foundation

final class SLMRuntime {
    private let modelDirectoryURL: URL?
    private let modelByteHint: Int

    init(modelDirectoryURL: URL?) {
        self.modelDirectoryURL = modelDirectoryURL
        self.modelByteHint = SLMRuntime.resolveModelByteHint(modelDirectoryURL)
    }

    func generate(prompt: String, maxTokens: Int = 32) -> String {
        let base = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "hello" : prompt
        let seed = (base.unicodeScalars.reduce(0) { $0 + Int($1.value) } + modelByteHint) % 9973
        let bounded = max(4, min(maxTokens, 24))

        var tokens: [String] = ["Echo:", base]
        for index in 0..<bounded {
            let token = (seed + (index * 37)) % 541
            tokens.append("tok\(token)")
        }
        return tokens.joined(separator: " ")
    }

    private static func resolveModelByteHint(_ modelDir: URL?) -> Int {
        guard let dir = modelDir else { return 0 }
        guard let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }

        var total = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) else {
                continue
            }
            if values.isRegularFile == true {
                total += values.fileSize ?? 0
            }
        }
        return total
    }
}
