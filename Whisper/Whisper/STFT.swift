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
    
    // MARK: FFT
    
    /// length of the FFT
    var fftLength:Int!

    /// FFT Window - should match one Frame of audio we process
    var fftWindowLength:Int!
    var fftWindowType:vDSP.WindowSequence!
    
    private var fft:ComplexFFTStandard!

    // MARK: STFT
    
    /// Total number of expected audio samples we process
    /// Our  sample count should be divisible by our fftLength / 2
    var sampleCount:Int!

    /// Number of samples we shift forward when constructing a new audio frame out of our input audio
    var hopCount:Int!
    
    
    // Calculate the number of iteractions we need to do
    // typically sampleCount / hopCount
    private var stftIterationCount:Int!
    
    
    init(fftLength:Int, windowType:vDSP.WindowSequence, windowLength:Int, sampleCount:Int, hopCount:Int  )
    {
        self.fftLength = fftLength
        self.fftWindowType = windowType
        self.fftWindowLength = windowLength
        self.fft = ComplexFFTStandard(numFFT: fftLength)
        
        self.sampleCount = sampleCount
        self.hopCount = hopCount
        self.stftIterationCount = self.sampleCount / self.hopCount
    }
    
    /// Calculate STFT and return matrix of real and imaginary components calculated
    public func calculateSTFT(audio:[Int16]) -> ([[Float]], [[Float]])
    {
        // Step 1
        assert(self.sampleCount == audio.count)

        var audioFloat:[Float] = [Float](repeating: 0, count: audio.count)
                
        vDSP.convertElements(of: audio, to: &audioFloat)
        // Audio now in Float, at Signed Int ranges - matches Pytorch Exactly

        vDSP.divide(audioFloat, 32768.0, result: &audioFloat)
        // Audio now in -1.0 to 1.0 Float ranges - matches Pytorch exactly

        // insert numFFT/2 samples before and numFFT/2 after so we have a extra numFFT amount to process
        audioFloat.insert(contentsOf: [Float](repeating: 0, count: self.fft.numFFT/2), at: 0)
        audioFloat.append(contentsOf: [Float](repeating: 0, count: self.fft.numFFT/2))
        
        // Alternatively all at the end?
//        audioFloat.append(contentsOf: [Float](repeating: 0, count: self.fft.numFFT))
        

        // Split Complex arrays holding the FFT results
        var allSampleReal:[[Float]] = []
        var allSampleImaginary:[[Float]] = []

        // Step 2 - we need to create 3000 x 200 matrix of windowed FFTs
        // Pytorch outputs complex numbers
        for (m) in 0 ..< self.stftIterationCount
        {
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            // audioFrame ends up holding split complex numbers
            
            // TODO: Handle Pytorch STFT Defaults:
            // TODO: Handle Centering  = True
            // TODO: Handle Padding = Reflect
            let audioFrame = Array<Float>( audioFloat[ (m * self.hopCount) ..< ( (m * self.hopCount) + self.fft.numFFT ) ] )

            assert(audioFrame.count == self.fft.numFFT)

            var (real, imaginary) = self.fft.forward(audioFrame)
            
            // For power spectrum, we ignore last 200 components
            if (real.count == self.fft.numFFT)
            {
                real = Array( real.prefix(upTo: self.fft.numFFT/2) )
                imaginary = Array( imaginary.prefix(upTo: self.fft.numFFT/2) )
            }


            assert(real.count == self.fft.numFFT / 2)
            
            allSampleReal.append(real)
            allSampleImaginary.append(imaginary)
        }
        
        return (allSampleReal, allSampleImaginary)
    }
    
}
