import Foundation
import Speech

guard CommandLine.arguments.count == 2 else {
    fputs("usage: transcribe_local.swift AUDIO_FILE\n", stderr)
    exit(2)
}

let audioURL = URL(fileURLWithPath: CommandLine.arguments[1])
let authSemaphore = DispatchSemaphore(value: 0)
var authorized = false

SFSpeechRecognizer.requestAuthorization { status in
    authorized = (status == .authorized)
    authSemaphore.signal()
}

_ = authSemaphore.wait(timeout: .now() + 15)
guard authorized else {
    fputs("Speech recognition is not authorized.\n", stderr)
    exit(3)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-GB")) else {
    fputs("Speech recognizer unavailable.\n", stderr)
    exit(4)
}

let request = SFSpeechURLRecognitionRequest(url: audioURL)
request.shouldReportPartialResults = false
request.requiresOnDeviceRecognition = false
if #available(macOS 13.0, *) {
    request.addsPunctuation = true
}

let resultSemaphore = DispatchSemaphore(value: 0)
var exitCode: Int32 = 0

let task = recognizer.recognitionTask(with: request) { result, error in
    if let result = result, result.isFinal {
        print(result.bestTranscription.formattedString)
        for segment in result.bestTranscription.segments {
            print(String(format: "%.3f\t%.3f\t%@", segment.timestamp, segment.duration, segment.substring))
        }
        resultSemaphore.signal()
    } else if let error = error {
        fputs("\(error.localizedDescription)\n", stderr)
        exitCode = 5
        resultSemaphore.signal()
    }
}

if resultSemaphore.wait(timeout: .now() + 90) == .timedOut {
    task.cancel()
    fputs("Speech recognition timed out.\n", stderr)
    exit(6)
}

exit(exitCode)
