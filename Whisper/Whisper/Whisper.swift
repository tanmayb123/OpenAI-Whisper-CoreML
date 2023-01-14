//
//  Whisper.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import Foundation
import CoreML
import AVFoundation
import Accelerate

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
    
   
//    let decoderModel: decoder
//    let encoderModel: encoder
    let tokenizer = WhisperTokenizer()
    
    let mel:MelSpectrogram = MelSpectrogram(sampleCount: kWhisperNumSamplesInChunk, hopCount: kWhisperHopLength, melCount: kWhisperNumMels, numFFT: kWhisperNumFFTs)

    // a chunk of audio samples, we decode that amount from some input
    // it seems like we pad by 200 in the beginning and end?
    
    var accruedAudioSamples:[Int16] = []
    var numOfAccruedAudioSamples:Int = 0
    
    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        
//        self.decoderModel = try decoder(configuration: config)
//        self.encoderModel = try encoder(configuration: config)
        
        self.accruedAudioSamples.reserveCapacity( Whisper.kWhisperNumSamplesInChunk )
    }
    
    func encode(audio: [Int16]) throws -> MLMultiArray {
        let mel:[Float] = mel.processData(audio: audio)
//        let mel = MelSpectrogram.loadReferencePythonRawMelToDebugShit()
        
        var normalizedFloatMel =  mel
        
        let count = normalizedFloatMel.count
        normalizedFloatMel.withUnsafeBufferPointer { unsafeMel in
            
//            var four:Float = 4.0
//            var eight:Float = 8.0
//            vDSP_vsadd(normalizedFloatMel, 1, &four, &normalizedFloatMel, 1, vDSP_Length(count))
//            vDSP_vsdiv(normalizedFloatMel, 1, &eight, &normalizedFloatMel, 1, vDSP_Length(count))
            
            
            let data = Data(buffer: unsafeMel)
            do {
                try data.write(to: URL(fileURLWithPath: "/Users/vade/Downloads/rawMel.raw"))
            }
            catch {
            }
        }
        
        let array = try MLMultiArray(shape: [1, 80, 3000], dataType: .float32)

        mel.withUnsafeBytes { melPtr in
            array.withUnsafeMutableBytes { arrayPtr, strides in
                memcpy(arrayPtr.baseAddress!, melPtr.baseAddress!, 80 * 3000 * MemoryLayout<Float>.size)
            }
        }
        
//        let encoded = try encoderModel.prediction(x_1:array).var_1373
//        return encoded
        return array
    }
    
    func decode(audioFeatures: MLMultiArray) throws {
        
//        // SOT Initialize sequence
//        var tokens:[Int] = []
//
//        // create sot sequence
//        // https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L325
//        // https://github.com/huggingface/transformers/blob/main/tests/models/whisper/test_tokenization_whisper.py
//        tokens.append(WhisperTokenizer.sotToken)
//        tokens.append(WhisperTokenizer.langToken)
//        tokens.append(WhisperTokenizer.transcribeToken)
////        tokens.append(WhisperTokenizer.notToken)
//        
//        var nextToken = 0
//        
//        while ( nextToken != WhisperTokenizer.eotToken )
//        {
//            autoreleasepool {
//                //            let transcription = self.tokenizer.decode(tokens: tokens)
//                //
//                //            print(transcription)
//                
//                let tokensArray = self.tokenizer.tokensToMultiArray(tokens, dims: 2)
//                //            let tokensArray = self.tokenizer.tokensToMultiArray([nextToken], dims: 2)
//                
//                let decoded = try! decoderModel.prediction(token_data: tokensArray, audio_data: audioFeatures).var_2205
//                
//                nextToken = self.tokenizer.nextTokenGreedy(decoded: decoded)
//                tokens.append(nextToken)
//                
//            }
//        }
//        
//        let transcription = self.tokenizer.decode(tokens: tokens)
//
//        print(transcription)

    }
    
    
    // this function accrues
    func accrueSamplesFromSampleBuffer(sampleBuffer:CMSampleBuffer)
    {
        
        var audioBufferListSize:Int = 0
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: &audioBufferListSize, bufferListOut: nil, bufferListSize:0, blockBufferAllocator: kCFAllocatorNull, blockBufferMemoryAllocator: kCFAllocatorNull, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: nil)
        
        var audioBufferList = AudioBufferList(mNumberBuffers: 1, mBuffers: AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(audioBufferListSize), mData: nil))

        var blockBuffer:CMBlockBuffer?
        
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, bufferListSizeNeededOut: nil, bufferListOut: &audioBufferList, bufferListSize: audioBufferListSize, blockBufferAllocator: kCFAllocatorNull, blockBufferMemoryAllocator: kCFAllocatorNull, flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment, blockBufferOut: &blockBuffer)
        
        // Determine the number of samples we need from our audio
        
        let numAvailableSamples = Int( CMSampleBufferGetNumSamples(sampleBuffer) )

        // Calculate the number of samples we have to acrrue to get a full chunk
        let remainingSampleCount = Whisper.kWhisperNumSamplesInChunk - self.accruedAudioSamples.count;
        
        let samplesToAccrue = min(numAvailableSamples, remainingSampleCount);
        
        let remainingCurrentSamplesInBuffer = numAvailableSamples - samplesToAccrue;
        
//        print("numAvailableSamples", numAvailableSamples, "samplesToAccrue", samplesToAccrue, "remainingSampleCount", remainingSampleCount)
        
        let unsafeAudioBufferList = UnsafeMutableAudioBufferListPointer(&audioBufferList)
        
        for (buffer) in unsafeAudioBufferList
        {
            let audioSampleArray:[Int16] = buffer.convertInt16()
                
            let samplesWeNeedToAccrueForAProperChunk = audioSampleArray[0 ... samplesToAccrue - 1]
            
            self.accruedAudioSamples.insert(contentsOf: samplesWeNeedToAccrueForAProperChunk, at: self.numOfAccruedAudioSamples)
                
            self.numOfAccruedAudioSamples = self.numOfAccruedAudioSamples + samplesWeNeedToAccrueForAProperChunk.count
            
            if (self.accruedAudioSamples.count == Whisper.kWhisperNumSamplesInChunk)
            {
                do {
                    let encoded = try self.encode(audio: self.accruedAudioSamples)
                    try self.decode(audioFeatures: encoded)
                }
                catch let error
                {
                    
                }
                self.accruedAudioSamples = []
                self.numOfAccruedAudioSamples = 0
            }
            
            
            if (remainingCurrentSamplesInBuffer > 0)
            {
                // Accrue whatever remainder we have..
                print("Remeber to Accrue left over samples")
            }
        }

    }
    
    
    func predict(assetURL:URL) async
    {
        let asset = AVURLAsset(url:assetURL)
        
        do {
            let assetReader = try AVAssetReader(asset: asset)
            
            let audioTracks = try await asset.loadTracks(withMediaType: .audio)
            
            // Output SInt 16
            let audioOutputSettings = [ AVFormatIDKey : kAudioFormatLinearPCM,
                                      AVSampleRateKey : 16000,
                                AVLinearPCMBitDepthKey: 16,
                                 AVNumberOfChannelsKey: 1,
                                AVLinearPCMIsFloatKey : false,
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
                guard let audioSampleBuffer = audioOutput.copyNextSampleBuffer(), CMSampleBufferIsValid(audioSampleBuffer) else {
                    
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
            
            let processingTime = NSDate.timeIntervalSinceReferenceDate - startTime
            
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

//extension AudioBufferList {
//    public mutating func convert() -> [AudioBuffer] {
//
//        self.mBuffers
//
//        let buf = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer<AudioBufferList>(start: &(self.mBuffers), count: Int(self.mNumberBuffers)) )
//
//        return Array(buf)
//
//
////        let buf: UnsafeBufferPointer<AudioBuffer> = UnsafeBufferPointer<AudioBuffer>(start: &(self.mBuffers), count: Int(self.mNumberBuffers))
////        return
//    }
//}

extension AudioBuffer {
    public func convertFloat() -> [Float] {
        if let mdata = self.mData {
            let ump = mdata.bindMemory(to: Float.self, capacity: Int(mDataByteSize))
            let usp = UnsafeBufferPointer(start: ump, count: Int(mDataByteSize) / MemoryLayout<Float>.size)
            return [Float](usp)
        } else {
            return []
        }
    }
    
    public func convertInt16() -> [Int16] {
        if let mdata = self.mData {
            let ump = mdata.bindMemory(to: Int16.self, capacity: Int(mDataByteSize))
            let usp = UnsafeBufferPointer(start: ump, count: Int(mDataByteSize) / MemoryLayout<Int16>.size)
            return [Int16](usp)
        } else {
            return []
        }
    }

}
