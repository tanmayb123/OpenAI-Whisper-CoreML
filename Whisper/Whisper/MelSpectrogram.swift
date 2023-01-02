//
//  stft.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//
import Accelerate



public class MelSpectrogram
{
    // MARK: Properties
    /// The number of audio samples per frame.
    static let sampleCount = 480000
    
    /// Determines the overlap between frames.
    static let hopCount = 160
    
    /// Number of displayed buffers — the width of the spectrogram.
    static let bufferCount = 3000
    
    /// The number of mel filter banks  — the height of the spectrogram.
    static let filterBankCount = 80
    
    static let ffTLength = 400
    
    /// Temporary buffers that the FFT operation uses for storing interim results.
    static var fftRealBuffer = [Float](repeating: 0, count: sampleCount / 2)
    static var fftImagBuffer = [Float](repeating: 0, count: sampleCount / 2)

    /// The forward fast Fourier transform object.
    static let fft: FFTSetup = {
//        let log2n = vDSP_Length(log2(Float(sampleCount)))
        let log2n = vDSP_Length(log2(Float(ffTLength)))

        guard let fft = vDSP_create_fftsetup(log2n,
                                             FFTRadix(kFFTRadix2)) else {
            fatalError("Unable to create FFT.")
        }
        
        return fft
    }()
    
    /// The window sequence used to reduce spectral leakage.
    static let hanningWindow = vDSP.window(ofType: Float.self,
                                           usingSequence: .hanningDenormalized,
                                           count: sampleCount,
                                           isHalfWindow: false)

    
    /// An array that contains the entire spectrogram.
    var melSpectrumValues = [Float](repeating: 0,
                                    count: bufferCount * filterBankCount)


    /// A reusable array that contains the current frame of time domain audio data as single-precision
    /// values.
    var timeDomainBuffer = [Float](repeating: 0,
                                   count: sampleCount)
    
    /// A resuable array that contains the frequency domain representation of the current frame of
    /// audio data.
    var frequencyDomainBuffer = [Float](repeating: 0,
                                        count: sampleCount)
    
    /// A matrix of `filterBankCount` rows and `sampleCount` that contains the triangular overlapping
    /// windows for each mel frequency.
    let filterBank = MelSpectrogram.makeFilterBankWithNumpyData()
    
    static let signalCount = 1
    /// A buffer that contains the matrix multiply result of the current frame of frequency domain values in
    /// `frequencyDomainBuffer` multiplied by the `filterBank` matrix.
    let sgemmResult = UnsafeMutableBufferPointer<Float>.allocate(capacity: MelSpectrogram.signalCount * Int(MelSpectrogram.filterBankCount))

    func processData(values: [Float]) {
 
        
        timeDomainBuffer = values
//        vDSP.convertElements(of: values,
//                             to: &timeDomainBuffer)

        MelSpectrogram.performForwardDFT(timeDomainValues: &timeDomainBuffer,
                                         frequencyDomainValues: &frequencyDomainBuffer,
                                         temporaryRealBuffer: &realParts,
                                         temporaryImaginaryBuffer: &imaginaryParts)
        
        vDSP.absolute(frequencyDomainBuffer,
                      result: &frequencyDomainBuffer)
        
        frequencyDomainBuffer.withUnsafeBufferPointer { frequencyDomainValuesPtr in
            cblas_sgemm(CblasRowMajor,
                        CblasTrans, CblasTrans,
                        Int32(MelSpectrogram.signalCount),
                        Int32(MelSpectrogram.filterBankCount),
                        Int32(MelSpectrogram.sampleCount),
                        1,
                        frequencyDomainValuesPtr.baseAddress, Int32(MelSpectrogram.signalCount),
                        filterBank.baseAddress, Int32(MelSpectrogram.sampleCount),
                        0,
                        sgemmResult.baseAddress, Int32(MelSpectrogram.filterBankCount))
        }
        
        vDSP_vdbcon(sgemmResult.baseAddress!, 1,
                    [20_000],
                    sgemmResult.baseAddress!, 1,
                    vDSP_Length(sgemmResult.count),
                    0)
 
        melSpectrumValues.append(contentsOf: sgemmResult)
    }
    
    /// The real parts of the time- and frequency-domain representations (the code performs DFT in-place)
    /// of the current frame of audio.
    var realParts = [Float](repeating: 0,
                            count: sampleCount / 2)
    
    /// The imaginary parts of the time- and frequency-domain representations (the code performs DFT
    /// in-place) of the current frame of audio.
    var imaginaryParts = [Float](repeating: 0,
                                 count: sampleCount / 2)
    
    /// Performs a forward Fourier transform on interleaved `timeDomainValues` writing the result to
    /// interleaved `frequencyDomainValues`.
    static func performForwardDFT(timeDomainValues: inout [Float],
                                  frequencyDomainValues: inout [Float],
                                  temporaryRealBuffer: inout [Float],
                                  temporaryImaginaryBuffer: inout [Float]) {
        
        vDSP.multiply(timeDomainValues,
                      hanningWindow,
                      result: &timeDomainValues)
        
        // Populate split real and imaginary arrays with the interleaved values
        // in `timeDomainValues`.
        temporaryRealBuffer.withUnsafeMutableBufferPointer { realPtr in
            temporaryImaginaryBuffer.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                   imagp: imagPtr.baseAddress!)
                
                timeDomainValues.withUnsafeBytes {
                    vDSP_ctoz($0.bindMemory(to: DSPComplex.self).baseAddress!, 2,
                              &splitComplex, 1,
                              vDSP_Length(MelSpectrogram.sampleCount / 2))
                }
            }
        }
        
        // Perform forward transform.
        temporaryRealBuffer.withUnsafeMutableBufferPointer { realPtr in
            temporaryImaginaryBuffer.withUnsafeMutableBufferPointer { imagPtr in
                fftRealBuffer.withUnsafeMutableBufferPointer { realBufferPtr in
                    fftImagBuffer.withUnsafeMutableBufferPointer { imagBufferPtr in
                        var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                           imagp: imagPtr.baseAddress!)
                        
                        var bufferSplitComplex = DSPSplitComplex(realp: realBufferPtr.baseAddress!,
                                                                 imagp: imagBufferPtr.baseAddress!)
                        
                        let log2n = vDSP_Length(log2(Float(sampleCount)))
                        
                        vDSP_fft_zript(fft,
                                       &splitComplex, 1,
                                       &bufferSplitComplex,
                                       log2n,
                                       FFTDirection(kFFTDirection_Forward))
                    }
                }
            }
        }
        
        // Populate interleaved `frequencyDomainValues` with the split values
        // from the real and imaginary arrays.
        temporaryRealBuffer.withUnsafeMutableBufferPointer { realPtr in
            temporaryImaginaryBuffer.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                   imagp: imagPtr.baseAddress!)
                
                frequencyDomainValues.withUnsafeMutableBytes { ptr in
                    vDSP_ztoc(&splitComplex, 1,
                              ptr.bindMemory(to: DSPComplex.self).baseAddress!, 2,
                              vDSP_Length(MelSpectrogram.sampleCount / 2))
                }
            }
        }
    }

    func generateSpectrogram(audio: [Float]) -> [Float] {
        
        processData(values: audio)
        
        return melSpectrumValues
    }


    static func makeFilterBankWithNumpyData() -> UnsafeMutableBufferPointer<Float> {
        let numpyFloatArrayLength = 16080
        let floatBytes = UnsafeMutableBufferPointer<Float>.allocate(capacity:numpyFloatArrayLength)
       
        do
        {
            let url = Bundle.main.url(forResource: "mel_filters", withExtension:"data")
            
            let numpyData = try Data(contentsOf: url!, options: [])
            
            _ = numpyData.copyBytes(to: floatBytes)
            
        }
        catch
        {
            
        }

        return floatBytes
    }
    
    
}

