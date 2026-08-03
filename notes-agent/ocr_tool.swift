// ocr_tool.swift — Vision framework OCR, called as subprocess by ocr_notes.py
// Usage: ./ocr_tool <image_path>
// Prints recognized text lines to stdout.

import Vision
import AppKit
import Foundation

guard CommandLine.arguments.count > 1 else {
    fputs("Usage: ocr_tool <image_path>\n", stderr)
    exit(1)
}

let imagePath = CommandLine.arguments[1]

guard let image = NSImage(contentsOfFile: imagePath),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    fputs("Failed to load image: \(imagePath)\n", stderr)
    exit(1)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("OCR failed: \(error)\n", stderr)
    exit(1)
}

if let results = request.results {
    for observation in results {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
}
