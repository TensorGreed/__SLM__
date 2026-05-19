import Foundation

let prompt = CommandLine.arguments.dropFirst().joined(separator: " ")
let base = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Hello from iOS" : prompt
let seed = base.unicodeScalars.reduce(0) { $0 + Int($1.value) } % 541
print("Echo: \(base) tok\(seed)")
