//
//  ContentView.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//

import SwiftUI
import AVFoundation

struct ContentView: View {
    let whisper: Whisper
    
#if os(iOS)

    @ObservedObject var recorder = AudioRecorder()

#elseif os(macOS)
    
    @ObservedObject var loader = AudioLoader()
    
#endif
    
    init() throws {
        whisper = try Whisper()
    }
    
    var body: some View {
#if os(iOS)

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
                
                do {
                    try recorder.startRecording()
                } catch let error {
                    fatalError("Couldn't record. Error: \(error)")
                }
                
                let audioAssetURL: URL
                do {
                    audioAssetURL = try recorder.finishRecording()
                } catch let error {
                    fatalError("Couldn't finish recording. Error: \(error)")
                }
                
                getAudioPredict(url: audioAssetURL)
            }
        }
        .onAppear {
            recorder.setup()
        }

#elseif os(macOS)
        ZStack {
            RoundedRectangle(cornerRadius: 20)
                .frame(height: 60)
                .padding()
                .foregroundColor(.blue)
            
            Text("Load Audio File")
                .font(.title2)
                .bold()
                .foregroundColor(.white)
        }
        .onTapGesture {
            
            getAudioPredict(url: loader.selectFileURL())
            
        }
#endif

    }

    func getAudioPredict(url:URL) {
     
        Task {
            do {
                let start = Date().timeIntervalSince1970
                
                await whisper.predict(assetURL: url)
                
                print(Date().timeIntervalSince1970 - start)
            } catch let error {
                fatalError("Couldn't predict. Error: \(error)")
            }
        }
    }
}
