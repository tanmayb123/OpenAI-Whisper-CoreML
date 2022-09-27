//
//  Whisper.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import Foundation
import CoreML

struct Whisper {
    static let LANGUAGES = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]
    
    let decoderModel: decoder
    let encoderModel: encoder
    
    init() throws {
        let config = MLModelConfiguration()
        self.decoderModel = try decoder(configuration: config)
        self.encoderModel = try encoder(configuration: config)
    }
    
    func encode(audio: [Double]) throws -> MLMultiArray {
        let spec = generateSpectrogram(audio: audio)
        let array = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)
        for (index, value) in spec.enumerated() {
            array[index] = NSNumber(floatLiteral: value)
        }
        let encoded = try encoderModel.prediction(x_1: array).var_1385
        return encoded
    }
    
    func decode(audioFeatures: MLMultiArray) throws {
        let sotToken = try MLMultiArray(shape: [1, 1], dataType: .float32)
        sotToken[0] = NSNumber(integerLiteral: 50258)
        let decoded = try decoderModel.prediction(x_1: sotToken, xa: audioFeatures).var_2217
        let confidence = (50259...50357).map { decoded[$0].floatValue }
        let (langIdx, _) = confidence.enumerated().max { $0.element < $1.element }!
        print(Self.LANGUAGES[langIdx])
    }
}
