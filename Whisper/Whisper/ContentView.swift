//
//  ContentView.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import SwiftUI

struct ContentView: View {
    let whisper: Whisper
    @ObservedObject var recorder = AudioRecorder()
    
    init() throws {
        whisper = try Whisper()
    }
    
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 20)
                .frame(height: 60)
                .padding()
                .foregroundColor(!recorder.canRecord || recorder.recording ? .gray : .blue)
            
            Text(!recorder.canRecord ? "Waiting for permissions..." : (recorder.recording ? "Recording..." : "Record"))
                .font(.title2)
                .bold()
                .foregroundColor(.white)
        }
        .onTapGesture {
            if recorder.canRecord && !recorder.recording {
                getAudioPredict()
            }
        }
        .onAppear {
            recorder.setup()
        }
    }
    
    func getAudioPredict() {
        do {
            try recorder.startRecording()
        } catch let error {
            fatalError("Couldn't record. Error: \(error)")
        }
        
        Timer.scheduledTimer(withTimeInterval: 10, repeats: false) { _ in
            let audio: [Float]
            do {
                audio = try recorder.finishRecording()
            } catch let error {
                fatalError("Couldn't finish recording. Error: \(error)")
            }
            
            do {
                let start = Date().timeIntervalSince1970
                var input = [Double](repeating: 0, count: 16000 * 30)
                for i in 0..<min(audio.count, input.count) {
                    input[i] = Double(audio[i])
                }
                let encoded = try whisper.encode(audio: input)
                try whisper.decode(audioFeatures: encoded)
                print(Date().timeIntervalSince1970 - start)
            } catch let error {
                fatalError("Couldn't predict. Error: \(error)")
            }
        }
    }
}
