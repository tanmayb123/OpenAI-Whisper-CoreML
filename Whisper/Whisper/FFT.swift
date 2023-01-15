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

    var window:[Double]
    var numFFT:Int = 400

    init(numFFT:Int)
    {
        self.numFFT = numFFT
        
        self.window = vDSP.window(ofType: Double.self,
                                         usingSequence: .hanningDenormalized,
                                         count: self.numFFT,
                                         isHalfWindow: false)
    }
}

/// Interleaves both the complex and imaginary components with the time domain audio signal
public class ComplexFFT : WhisperFFT
{
    var fft : vDSP.FFT<DSPDoubleSplitComplex>
    
    override init(numFFT: Int) {
        
        let log2n = vDSP_Length(log2(Float(numFFT)))

        self.fft = vDSP.FFT(log2n: log2n,
                           radix: .radix2,
                           ofType: DSPDoubleSplitComplex.self)!

        super.init(numFFT:numFFT)
    }
    
    public func forward(_ audioFrame:[Double]) -> ([Double], [Double])
    {
        var sampleReal:[Double] = [Double](repeating: 0, count: self.numFFT/2)
        var sampleImaginary:[Double] = [Double](repeating: 0, count: self.numFFT/2)

        var resultReal:[Double] = [Double](repeating: 0, count: self.numFFT/2)
        var resultImaginary:[Double] = [Double](repeating: 0, count: self.numFFT/2)
        
        var windowedAudioFrame = [Double](repeating: 0, count: self.numFFT)
        
        vDSP.multiply(audioFrame,
                      self.window,
                      result: &windowedAudioFrame)

        sampleReal.withUnsafeMutableBytes { unsafeReal in
            sampleImaginary.withUnsafeMutableBytes { unsafeImaginary in

                resultReal.withUnsafeMutableBytes { unsafeResultReal in
                    resultImaginary.withUnsafeMutableBytes { unsafeResultImaginary in
                        
                        
                        var complexSignal = DSPDoubleSplitComplex(realp: unsafeReal.bindMemory(to: Double.self).baseAddress!,
                                                                  imagp: unsafeImaginary.bindMemory(to: Double.self).baseAddress!)
                        
                        var complexResult = DSPDoubleSplitComplex(realp: unsafeResultReal.bindMemory(to: Double.self).baseAddress!,
                                                                  imagp: unsafeResultImaginary.bindMemory(to: Double.self).baseAddress!)
                        
                        windowedAudioFrame.withUnsafeBytes { unsafeAudioBytes in
                            vDSP.convert(interleavedComplexVector: [DSPDoubleComplex](unsafeAudioBytes.bindMemory(to: DSPDoubleComplex.self)),
                                         toSplitComplexVector: &complexSignal)
                        }
                        
                        // Step 3 - creating the FFT
                        self.fft.transform(input: complexSignal, output: &complexResult, direction: vDSP.FourierTransformDirection.forward)
                    
                        // Scale by 1/2 : https://stackoverflow.com/questions/51804365/why-is-fft-different-in-swift-than-in-python
                        var scaleFactor = Double( 1.0/2.0 ) // * 1.165 ??
                        vDSP_vsmulD(complexResult.realp, 1, &scaleFactor, complexResult.realp, 1, vDSP_Length(self.numFFT/2))
                        vDSP_vsmulD(complexResult.imagp, 1, &scaleFactor, complexResult.imagp, 1, vDSP_Length(self.numFFT/2))
                    }
                }
            }
        }

        return (resultReal, resultImaginary)
    }
}

/// Populates only the real components with the time domain audio signal
// https://stackoverflow.com/questions/51804365/why-is-fft-different-in-swift-than-in-python
public class RealFFT : WhisperFFT
{

    public func forward(_ audioFrame:[Double]) -> ([Double], [Double])
    {
        let input_windowed = vDSP.multiply(audioFrame, self.window)
        
        var real = [Double](input_windowed[0 ..< input_windowed.count])
        var imaginary = [Double](repeating: 0.0, count: input_windowed.count)

        real.withUnsafeMutableBufferPointer { realBuffer in
            imaginary.withUnsafeMutableBufferPointer { imaginaryBuffer in
                
                var splitComplex = DSPDoubleSplitComplex(
                    realp: realBuffer.baseAddress!,
                    imagp: imaginaryBuffer.baseAddress!
                )
                
                let length = vDSP_Length(floor(log2(Float(input_windowed.count))))
                let radix = FFTRadix(kFFTRadix2)
                let weights = vDSP_create_fftsetupD(length, radix)
                withUnsafeMutablePointer(to: &splitComplex) { splitComplex in
                    vDSP_fft_zipD(weights!, splitComplex, 1, length, FFTDirection(FFT_FORWARD))
                    
                }
                
                
//                // zvmags yields power spectrum: |S|^2
//                withUnsafePointer(to: &splitComplex) { splitComplex in
//                    magnitudes.withUnsafeMutableBufferPointer { magnitudes in
//                        vDSP_zvmags(splitComplex, 1, magnitudes.baseAddress!, 1, vDSP_Length(input_windowed.count))
//                    }
//                }
//
//                vDSP_destroy_fftsetup(weights)
            }
        }

        return (real, imaginary)
    }
    
}
