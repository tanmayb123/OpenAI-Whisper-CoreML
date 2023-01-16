//
//  STFT.swift
//  Whisper
//
//  Created by Anton Marini on 1/14/23.
//

import Foundation
import Accelerate

class STFT
{
    /// An attempt at mimiking Torch STFT
    /// Consume 'sampleCount' SInt16 single channel audio buffers
    /// Produce a complex STFT output
    ///
    /// Note, audioFrames we produce are not padded, or reflected or centered
    ///
    
    enum Padding {
        case Reflect
        case Zero
//        case
    }
    
    
    // MARK: FFT
    
    /// length of the FFT
    var fftLength:Int!

    /// FFT Window - should match one Frame of audio we process
    var fftWindowLength:Int!
    var fftWindowType:vDSP.WindowSequence!
    
    private var fft:RealFFT!

    // MARK: STFT
    
    /// Total number of expected audio samples we process
    /// Our  sample count should be divisible by our fftLength / 2
    var sampleCount:Int!

    /// Number of samples we shift forward when constructing a new audio frame out of our input audio
    var hopCount:Int!
    
    var padding:Padding!
    var center:Bool!
    
    // Calculate the number of iteractions we need to do
    // typically sampleCount / hopCount
    private var stftIterationCount:Int!
    
    
    init(fftLength:Int, windowType:vDSP.WindowSequence, windowLength:Int, sampleCount:Int, hopCount:Int, center:Bool = true, padding:Padding = .Reflect )
    {
        self.fft = RealFFT(numFFT: fftLength)

        self.fftLength = fftLength
        self.fftWindowType = windowType
        self.fftWindowLength = windowLength
        
        
        self.sampleCount = sampleCount
        self.hopCount = hopCount
        self.stftIterationCount = self.sampleCount / self.hopCount
        
        self.padding = padding
        self.center = center
    }
    
    /// Calculate STFT and return matrix of real and imaginary components calculated
    public func calculateSTFT(audio:[Int16]) -> ([[Double]], [[Double]], [[Double]])
    {
        // Step 1
        assert(self.sampleCount == audio.count)

        var audioFloat:[Double] = [Double](repeating: 0, count: audio.count)
                
        vDSP.convertElements(of: audio, to: &audioFloat)
        // Audio now in Float, at Signed Int ranges - matches Pytorch Exactly

        vDSP.divide(audioFloat, 32768.0, result: &audioFloat)
        // Audio now in -1.0 to 1.0 Float ranges - matches Pytorch exactly

        // Center pad, reflect mode
        
        if (self.center)
        {
            switch ( self.padding )
            {
            case .Reflect, .none:
                let reflectStart = audioFloat[0 ..< self.fftLength/2]
                let reflectEnd = audioFloat[audioFloat.count -  self.fftLength/2 ..< audioFloat.count]
                
                audioFloat.insert(contentsOf:reflectStart.reversed(), at: 0)
                audioFloat.append(contentsOf:reflectEnd.reversed())
            case .Zero:
                let zero:[Double] = [Double](repeating: 0, count: self.fftLength/2 )
                
                audioFloat.insert(contentsOf:zero, at: 0)
                audioFloat.append(contentsOf:zero)
            }
        }
        else
        {
            // Alternatively all at the end?
            audioFloat.append(contentsOf: [Double](repeating: 0, count: self.fftLength))
        }
        // Split Complex arrays holding the FFT results
        var allSampleReal:[[Double]] = []
        var allSampleImaginary:[[Double]] = []
        var allSampleMagnitudes:[[Double]] = []

        // Step 2 - we need to create 3000 x 200 matrix of windowed FFTs
        // Pytorch outputs complex numbers
        for (m) in 0 ..< self.stftIterationCount
        {
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            // audioFrame ends up holding split complex numbers
            
            // TODO: Handle Pytorch STFT Defaults:
            // TODO: Handle Centering  = True
            // TODO: Handle Padding = Reflect
            let audioFrame = Array<Double>( audioFloat[ (m * self.hopCount) ..< ( (m * self.hopCount) + self.fftLength ) ] )

            assert(audioFrame.count == self.fftLength)
            
            var (real, imaginary, magnitudes) = self.fft.forward(audioFrame)
            
            // We divide our half our FFT output,
            // because the Pytorch `onesized` is true by default for real valued signals
            // See https://pytorch.org/docs/stable/generated/torch.stft.html
            
            if (real.count == self.fftLength )
            {
                real = Array(real.prefix(upTo:1 + self.fftLength / 2))
                imaginary = Array(imaginary.prefix(upTo:1 + self.fftLength / 2))
                magnitudes = Array(magnitudes.prefix(upTo:1 + self.fftLength / 2))

            }
            
            assert(real.count == 1 + self.fft.numFFT / 2)
            assert(imaginary.count == 1 +  self.fft.numFFT / 2)

            allSampleReal.append(real)
            allSampleImaginary.append(imaginary)
            allSampleMagnitudes.append(magnitudes)
        }
        
        return (allSampleReal, allSampleImaginary, allSampleMagnitudes)
    }
    
}
