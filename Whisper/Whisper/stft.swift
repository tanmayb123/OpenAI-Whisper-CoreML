//
//  stft.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

func generateSpectrogram(audio: [Double]) -> [Double] {
    var audio = audio
    audio.insert(contentsOf: [Double](repeating: 0, count: 200), at: 0)
    audio.append(contentsOf: [Double](repeating: 0, count: 200))
    var result = [Double](repeating: 0, count: 80 * 3000)
    audio.withUnsafeMutableBufferPointer { audioPtr in
        result.withUnsafeMutableBufferPointer { resultPtr in
            generate_spectrogram(audioPtr.baseAddress!, resultPtr.baseAddress!)
        }
    }
    return result
}
