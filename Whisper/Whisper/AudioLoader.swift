//
//  AudioLoader.swift
//  Whisper
//
//  Created by vade on 1/2/23.
//

import Foundation
import Cocoa

class AudioLoader: NSObject, ObservableObject
{
    func selectFileURL() -> URL {
        
        let openpanel = NSOpenPanel()
        
        openpanel.runModal()
        
        return openpanel.url!
    }
}
