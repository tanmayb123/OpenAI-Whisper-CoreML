//
//  Whisper.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import Foundation
import CoreML
import AVFoundation

public class Whisper {
    
    // hard-coded audio hyperparameters
    static let kWhisperSampleRate:Int = 16000;
    static let kWhisperNumFFTs:Int = 400;
    static let kWhisperNumMels:Int = 80;
    static let kWhisperHopLength:Int = 160;
    static let kWhisperChunkTimeSeconds:Int = 30;
    // kWhisperChunkTimeSeconds * kWhisperSampleRate  # 480000: number of samples in a chunk
    static let kWhisperNumSamplesInChunk:Int = 480000; // Raw audio chunks we convert to MEL
    // exact_div(kWhisperNumSamplesInChunk, kWhisperHopLength)  # 3000: number of frames in a mel spectrogram input
    static let kWhisperNumSamplesInMel:Int = 3000; // frames of Mel spectrograms
    
    static let LANGUAGES = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "iw", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]
    
//    let decoderModel: decoder
//    let encoderModel: encoder
    let mel:MelSpectrogram = MelSpectrogram()

    // a chunk of audio samples, we decode that amount from some input
    // it seems like we pad by 200 in the beginning and end?
    var accruedAudioSamples:[Float] = []
    var numOfAccruedAudioSamples:Int = 0
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
//        self.decoderModel = try decoder(configuration: config)
//        self.encoderModel = try encoder(configuration: config)
        
        self.accruedAudioSamples.reserveCapacity( Whisper.kWhisperNumSamplesInChunk )
//        self.accruedAudioSamples.append(contentsOf: [Float](repeating: 0, count: 200))
    }
    
    func encode(audio: [Float]) throws -> MLMultiArray {
        mel.processData(values: audio)

        let spec = mel.melSpectrumValues
        let array = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)

        for (index, value) in spec.enumerated() {
            array[index] = NSNumber(value: value)
        }

//        let encoded = try encoderModel.prediction(audio_input:array).var_1373
        return array
    }
    
    func decode(audioFeatures: MLMultiArray) throws {
        let sotToken = try MLMultiArray(shape: [1, 1], dataType: .float32)
        sotToken[0] = NSNumber(integerLiteral: 50258)
//        let decoded = try decoderModel.prediction(token_data: sotToken, audio_data: audioFeatures).var_2205
//        let confidence = (50259...50357).map { decoded[$0].floatValue }
//        let (langIdx, _) = confidence.enumerated().max { $0.element < $1.element }!
//        print(Self.LANGUAGES[langIdx])
    }
    
    
    // this function accrues
    func accrueSamplesFromSampleBuffer(sampleBuffer:CMSampleBuffer)
    {
        var audioBufferListSize:Int = 0
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: &audioBufferListSize, bufferListOut: nil, bufferListSize:0, blockBufferAllocator: kCFAllocatorDefault, blockBufferMemoryAllocator: kCFAllocatorDefault, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: nil)
        
        var audioBufferList = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(audioBufferListSize), mData: nil))

        var blockBuffer:CMBlockBuffer?
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &audioBufferList, bufferListSize: audioBufferListSize, blockBufferAllocator: kCFAllocatorDefault, blockBufferMemoryAllocator: kCFAllocatorDefault, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: &blockBuffer)
        
        // Determine the number of samples we need from our audio
        
        let numAvailableSamples = Int( CMSampleBufferGetNumSamples(sampleBuffer) )

        // Calculate the number of samples we have to acrrue to get a full chunk
        let remainingSampleCount = Whisper.kWhisperNumSamplesInChunk - self.accruedAudioSamples.count;
        
        let samplesToAccrue = min(numAvailableSamples, remainingSampleCount);
        
        let remainingCurrentSamplesInBuffer = numAvailableSamples - samplesToAccrue;
        
        print("numAvailableSamples", numAvailableSamples, "samplesToAccrue", samplesToAccrue, "remainingSampleCount", remainingSampleCount)
        
                        
        for (buffer) in audioBufferList.convert()
        {
            let floatArray:[Float] = buffer.convert()
                
            let samplesWeNeedToAccrueForAProperChunk = floatArray[0 ... samplesToAccrue - 1]
            
            self.accruedAudioSamples.insert(contentsOf: samplesWeNeedToAccrueForAProperChunk, at: self.numOfAccruedAudioSamples)
                
            self.numOfAccruedAudioSamples = self.numOfAccruedAudioSamples + samplesWeNeedToAccrueForAProperChunk.count
            
            if (self.accruedAudioSamples.count == Whisper.kWhisperNumSamplesInChunk)
            {
                
                print("Sending Chunk to Mel")
    //            self.accruedAudioSamples.append(contentsOf: [Float](repeating: 0, count: 200))
                
                // send to Mel
                self.mel.processData(values: self.accruedAudioSamples)
                
                
                self.accruedAudioSamples = []
                self.numOfAccruedAudioSamples = 0
            }
            
            
            if (remainingCurrentSamplesInBuffer > 0)
            {
                // Accrue whatever remainder we have..
                print("Accrue")
            }
        }
    
        
   
        

    }
    
    
    func predict(assetURL:URL) async
    {
        let asset = AVURLAsset(url:assetURL)
        
        do {
            let assetReader = try AVAssetReader(asset: asset)
            
            let audioTracks = try await asset.loadTracks(withMediaType: .audio)
            
            let audioOutputSettings = [ AVFormatIDKey : kAudioFormatLinearPCM,
                                      AVSampleRateKey : 16000,
                                AVLinearPCMBitDepthKey: 32,
                                 AVNumberOfChannelsKey: 1,
                                AVLinearPCMIsFloatKey : true,
                           AVLinearPCMIsNonInterleaved: false,
                             AVLinearPCMIsBigEndianKey: false
                                        
            ] as [String : Any]
            
            let audioOutput = AVAssetReaderAudioMixOutput(audioTracks: audioTracks, audioSettings: audioOutputSettings)
            audioOutput.alwaysCopiesSampleData = false
            
            if ( assetReader.canAdd(audioOutput) )
            {
                assetReader.add(audioOutput)
            }
            
            assetReader.startReading()
            
            let startTime = NSDate.timeIntervalSinceReferenceDate
            
            while ( assetReader.status == .reading )
            {
                guard let audioSampleBuffer = audioOutput.copyNextSampleBuffer() else {
                    
                    // Some media formats can have weird decode issues.
                    // Unless our asset reader EXPLICITELT tells us its done, keep trying to decode.
                    // We just skip bad samples
                    if ( assetReader.status == .reading)
                    {
                        continue
                    }
                    
                    else if (assetReader.status == .completed)
                    {
                        break;
                    }
                    
                    else
                    {
                        // something went wrong
                        print(assetReader.error as Any)
                        return
                    }
                        
                }
                                        
                self.accrueSamplesFromSampleBuffer(sampleBuffer: audioSampleBuffer)
                
            }
            
            let processingTime = startTime - NSDate.timeIntervalSinceReferenceDate
            
            print("Decode and Predict took", processingTime, "seconds")
            
            let assetDuration = try await asset.load(.duration).seconds
            
            print("Movie is", assetDuration)
            print("Realtime Factor is", assetDuration / processingTime)

        }
        catch let error
        {
            print("Unable to process asset:")
            print(error)
            exit(0)
        }
    }
    
}


// Taken from : https://gist.github.com/tion-low/47e9fc4082717078dff4d6259b6ffbc9

extension AudioBufferList {
    public mutating func convert() -> [AudioBuffer] {
        let buf: UnsafeBufferPointer<AudioBuffer> = UnsafeBufferPointer<AudioBuffer>(start: &(self.mBuffers), count: Int(self.mNumberBuffers))
        return Array(buf)
    }
}

extension AudioBuffer {
    public func convert() -> [Float] {
        if let mdata = self.mData {
            let ump = mdata.bindMemory(to: Float.self, capacity: Int(mDataByteSize))
            let usp = UnsafeBufferPointer(start: ump, count: Int(mDataByteSize) / MemoryLayout<Float>.size)
            return [Float](usp)
        } else {
            return []
        }
    }
}
