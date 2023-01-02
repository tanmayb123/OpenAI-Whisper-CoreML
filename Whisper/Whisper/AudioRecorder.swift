//
//  AudioRecorder.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-27.
//

import AVFoundation
import SwiftUI

func getDocumentsDirectory() -> URL {
    let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
    return paths[0]
}

class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    static let audioURL = getDocumentsDirectory().appendingPathComponent("query.wav")
    
    private var recordingSession: AVAudioSession!
    private var audioRecorder: AVAudioRecorder!
    
    @Published private(set) var canRecord = false
    @Published private(set) var recording = false
    
    enum RecordingError: Error {
        case invalidFormat
        case noBuffer
    }
    
    func setup() {
        recordingSession = AVAudioSession.sharedInstance()
        do {
            try recordingSession.setCategory(.record)
            try recordingSession.setMode(.measurement)
        } catch let error {
            print(error)
        }
        do {
            try recordingSession.setCategory(.playAndRecord, mode: .default)
            try recordingSession.setActive(true)
            recordingSession.requestRecordPermission { allowed in
                DispatchQueue.main.async {
                    if allowed {
                        self.canRecord = true
                    } else {
                        fatalError("User did not allow access to microphone")
                    }
                }
            }
        } catch let error {
            fatalError("\(error)")
        }
    }
    
    func startRecording() throws {
        let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        audioRecorder = try AVAudioRecorder(url: Self.audioURL, settings: settings)
        audioRecorder.delegate = self
        audioRecorder.record()
        recording = true
    }
    
    func finishRecording() throws -> URL {
        audioRecorder.stop()
        audioRecorder = nil
        recording = false
        
        return AudioRecorder.audioURL
    }
    
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if !flag {
            fatalError("Error in recording")
        }
    }
}
