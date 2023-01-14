//
//  FFTImplementations.swift
//  Whisper
//
//  Created by Anton Marini on 1/12/23.
//

import Foundation
import Accelerate

public class WhisperFFT
{

    var window:[Float]
    var numFFT:Int = 400

    init(numFFT:Int)
    {
        self.numFFT = numFFT
        
        self.window = vDSP.window(ofType: Float.self,
                                         usingSequence: .hanningDenormalized,
                                         count: self.numFFT,
                                         isHalfWindow: false)
    }
}

public class ComplexFFTStandard : WhisperFFT
{
    var fft : vDSP.FFT<DSPSplitComplex>
    
    override init(numFFT: Int) {
        
        let log2n = vDSP_Length(log2(Float(numFFT)))

        self.fft = vDSP.FFT(log2n: log2n,
                           radix: .radix2,
                           ofType: DSPSplitComplex.self)!

        super.init(numFFT:numFFT)
    }
    
    public func forward(_ audioFrame:[Float]) -> ([Float], [Float])
    {
        var sampleReal:[Float] = [Float](repeating: 0, count: self.numFFT/2)
        var sampleImaginary:[Float] = [Float](repeating: 0, count: self.numFFT/2)

        var resultReal:[Float] = [Float](repeating: 0, count: self.numFFT/2)
        var resultImaginary:[Float] = [Float](repeating: 0, count: self.numFFT/2)
        
        var windowedAudioFrame = [Float](repeating: 0, count: self.numFFT)
        
        sampleReal.withUnsafeMutableBytes { unsafeReal in
            sampleImaginary.withUnsafeMutableBytes { unsafeImaginary in

                resultReal.withUnsafeMutableBytes { unsafeResultReal in
                    resultImaginary.withUnsafeMutableBytes { unsafeResultImaginary in
                        
                        vDSP.multiply(audioFrame,
                                      self.window,
                                      result: &windowedAudioFrame)
                        
                        var complexSignal = DSPSplitComplex(realp: unsafeReal.bindMemory(to: Float.self).baseAddress!,
                                                            imagp: unsafeImaginary.bindMemory(to: Float.self).baseAddress!)
                        
                        var complexResult = DSPSplitComplex(realp: unsafeResultReal.bindMemory(to: Float.self).baseAddress!,
                                                            imagp: unsafeResultImaginary.bindMemory(to: Float.self).baseAddress!)
                        
                        windowedAudioFrame.withUnsafeBytes { unsafeAudioBytes in
                            vDSP.convert(interleavedComplexVector: [DSPComplex](unsafeAudioBytes.bindMemory(to: DSPComplex.self)),
                                         toSplitComplexVector: &complexSignal)
                        }
                        
                        // Step 3 - creating the FFT
                        self.fft.forward(input: complexSignal, output: &complexResult)
                    }
                }
            }
        }

        return (resultReal, resultImaginary)
    }
}


public class PowerSpectrumComplexInterleaved : WhisperFFT
{
    public func forward(_ audioFrame:[Float]) -> ([Float], [Float])
    {
        let input_windowed = vDSP.multiply(audioFrame, self.window)
        
        var inputReal = [Float](repeating: 0.0, count: input_windowed.count / 2)
        var inputImaginary = [Float](repeating: 0.0, count: input_windowed.count / 2)
        
        var resultReal = [Float](repeating: 0.0, count: input_windowed.count / 2)
        var resultImaginary = [Float](repeating: 0.0, count: input_windowed.count / 2)
        
        inputReal.withUnsafeMutableBufferPointer { realBuffer in
            inputImaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                
                resultReal.withUnsafeMutableBufferPointer { resultRealBuffer in
                    resultImaginary.withUnsafeMutableBufferPointer { resultImaginaryBuffer in
                        
                        var inputSplitComplex = DSPSplitComplex(
                            realp: realBuffer.baseAddress!,
                            imagp: imaginaryBuffer.baseAddress!
                        )
                        
                        
                        input_windowed.withUnsafeBytes {
                            vDSP_ctoz($0.bindMemory(to: DSPComplex.self).baseAddress!, 2,
                                      &inputSplitComplex, 1,
                                      vDSP_Length(input_windowed.count / 2))
                        }
                        
                        var resultSplitComplex = DSPSplitComplex(
                            realp: resultRealBuffer.baseAddress!,
                            imagp: resultImaginaryBuffer.baseAddress!
                        )
                        
                        
                        let length = vDSP_Length(floor(log2(Float(input_windowed.count / 2))))
                        let radix = FFTRadix(kFFTRadix2)
                        let weights = vDSP_create_fftsetup(length, radix)
                        withUnsafeMutablePointer(to: &inputSplitComplex) { inputSplitComplex in
                            withUnsafeMutablePointer(to: &resultSplitComplex) { resultSplitComplex in
                                
                                //                    vDSP_fft_zip(weights!, splitComplex, 1, length, FFTDirection(FFT_FORWARD))
                                vDSP_fft_zrop(weights!, inputSplitComplex, 1, resultSplitComplex, 1, length, FFTDirection(FFT_FORWARD))
                            }
                            
                            vDSP_destroy_fftsetup(weights)
                            
                            
                        }
                    }
                }
            }
        }
        
        return (resultReal, resultImaginary)
    }
}

// Taken from Surge FFT discussion
public class PowerSpectrum : WhisperFFT
{

    public func forward(_ audioFrame:[Float]) -> ([Float], [Float])
    {
        // window has slightly different precision than scipy.window.hamming (float vs double)
        var win = [Float](repeating: 0, count: audioFrame.count)
        vDSP_hann_window(&win, vDSP_Length(audioFrame.count), Int32(vDSP_HANN_DENORM))
        
        let input_windowed = vDSP.multiply(audioFrame, win)
        
        var real = [Float](input_windowed[0 ..< input_windowed.count])
        var imaginary = [Float](repeating: 0.0, count: input_windowed.count)

//        var magnitudes = [Float](repeating: 0.0, count: input_windowed.count)

        real.withUnsafeMutableBufferPointer { realBuffer in
            imaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                
                var splitComplex = DSPSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                
                let length = vDSP_Length(floor(log2(Float(input_windowed.count))))
                let radix = FFTRadix(kFFTRadix2)
                let weights = vDSP_create_fftsetup(length, radix)
                withUnsafeMutablePointer(to: &splitComplex) { splitComplex in
                    vDSP_fft_zip(weights!, splitComplex, 1, length, FFTDirection(FFT_FORWARD))
                }
                
//                withUnsafePointer(to: &splitComplex) { splitComplex in
//                    magnitudes.withUnsafeMutableBufferPointer { magnitudes in
//                        // zvmags yields power spectrum: |S|^2
//                        vDSP_zvmags(splitComplex, 1, magnitudes.baseAddress!, 1, vDSP_Length(input_windowed.count))
//                    }
//                }
                
                vDSP_destroy_fftsetup(weights)
            }
        }

        return (real, imaginary)
    }
    
}
