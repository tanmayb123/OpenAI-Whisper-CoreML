//
//  stft.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//
import Accelerate
import Numerics



// Look at
// https://github.com/openai/whisper/blob/main/whisper/audio.py#L92
/*
  window = torch.hann_window(N_FFT).to(audio.device)
  stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
 
  magnitudes = stft[:, :-1].abs() ** 2

  filters = mel_filters(audio.device, n_mels)
  mel_spec = filters @ magnitudes

  log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
  log_spec = (log_spec + 4.0) / 4.0
 
 stft torch.Size([201, 3001])
 magnitudes torch.Size([201, 3000])
 mel filters torch.Size([80, 201])
 mel spec torch.Size([80, 3000])
 log spec torch.Size([80, 3000])
 
 */

// https://pytorch.org/docs/stable/generated/torch.stft.html
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L820



// alternatively http://www.ml-illustrated.com/2020/06/01/deploy-pytorch-model-with-coreml-convert-issues.html
// and https://github.com/ml-illustrated/Pytorch-CoreML-Spectrogram/blob/d0dd6c55eaf5fdcfaf00b1f036b258bd144b1ac4/python/model.py#L142

public class MelSpectrogram
{
    // MARK: Properties
    
    /// An 2D  array that contains the entire spectrogram, 80 x 3000
    var melSpectrumValues:[[Float]]
    
    // a 3000 x 201 array - we compute 201 complex numbers for each STFT we compute
    // this is transposed to 201 x 3000 and matix multiplies by our 80 x 201 mel filter matrix
    //var complexSpectrumValues:[[Complex<Float>]]

    /// A matrix of `filterBankCount` rows and `sampleCount` that contains the triangular overlapping
    /// windows for each mel frequency.
    var melFilterBank:UnsafeMutableBufferPointer<Float>

    /// Tthe width of the spectrogram.
    var melSampleCount:Int = 3000
    
    /// The height of the spectrogram.
    var melFilterBankCount:Int = 80

    /// The number of audio samples per chunk.
    var sampleCount:Int = 480000
    
    /// Determines the overlap between samples for an FFT.
    var hopCount:Int = 160

    var numFFT:Int = 400

    /// The forward fast Fourier transform object.
    var fft: FFTSetup

    
    // Variables below are used to calculate a mel spectrum / single row of our spectrogram
    
    /// The window sequence used to reduce spectral leakage.
    var hanningWindow:[Float]

    /// Temporary buffers that the FFT operation uses for storing interim results.
    var fftRealBuffer:[Float]
    var fftImagBuffer:[Float]

    /// A resuable array that contains the time domain representation of the current frame of
    /// audio data.
    var timeDomainValues:[Float]
    
    /// A resuable array that contains the frequency domain representation of the current frame of
    /// audio data.
    var frequencyDomainBuffer:[Float]

    /// A buffer that contains the matrix multiply result of the current frame of frequency domain values in
    /// `frequencyDomainBuffer` multiplied by the `filterBank` matrix.
    var sgemmResult:[Float] // UnsafeMutableBufferPointer<Float>

    /// The real parts of the time- and frequency-domain representations (the code performs DFT in-place)
    /// of the current frame of audio.
    var realParts:[Float]
    
    /// The imaginary parts of the time- and frequency-domain representations (the code performs DFT
    /// in-place) of the current frame of audio.
    var imaginaryParts:[Float]

    init(sampleCount:Int, hopCount:Int, melCount:Int, numFFT:Int)
    {
        self.sampleCount = sampleCount
        self.hopCount = hopCount
        self.melFilterBankCount = melCount
        
        self.melSampleCount = self.sampleCount / self.hopCount
        
        self.numFFT = numFFT
                
        self.hanningWindow = vDSP.window(ofType: Float.self,
                                         usingSequence: .hanningDenormalized,
                                         count: self.numFFT,
                                         isHalfWindow: false)
        
        self.melSpectrumValues = [[Float]](repeating: [Float](repeating: 0, count: self.melSampleCount), count: self.melFilterBankCount  )
        
        self.timeDomainValues = [] //[Float](repeating: 0, count: self.melSampleCount)
        self.frequencyDomainBuffer = [Float](repeating: 0, count: self.numFFT)
        
        self.sgemmResult = [Float](repeating: 0, count: self.melSampleCount);//UnsafeMutableBufferPointer<Float>.allocate(capacity: self.melSampleCount)
        
        self.realParts = [Float](repeating: 0,count: self.numFFT / 2)
        self.imaginaryParts = [Float](repeating: 0,count: self.numFFT / 2)
        self.fftRealBuffer = [Float](repeating: 0, count: self.numFFT / 2)
        self.fftImagBuffer = [Float](repeating: 0, count: self.numFFT / 2)
        
        self.fft = {
            let log2n = vDSP_Length(log2(Float(sampleCount)))
            
            guard let fft = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
                fatalError("Unable to create FFT.")
            }
            
            return fft
        }()
        
        self.melFilterBank = MelSpectrogram.makeFilterBankWithNumpyData()
    }
    
    func processData(audio: [Float])
    {
        assert(self.sampleCount == audio.count)
     
        // insert numFFT/2 samples before and numFFT/2 after so we have a extra numFFT amount to process
        var audio = audio
        audio.insert(contentsOf: [Float](repeating: 0, count: self.numFFT/2), at: 0)
        audio.append(contentsOf: [Float](repeating: 0, count: self.numFFT/2))

        // we need to create 201 x 3001 matrix of STFTs - note we appear to want to output complex numbers (?)
        
        for (i) in 0 ..< self.melSampleCount
        {
            print("working on mel sample", i)
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            self.timeDomainValues = Array<Float>( audio[ (i * self.hopCount) ..< ( (i * self.hopCount) + self.numFFT) ] )

            assert(self.timeDomainValues.count == self.numFFT)
            
            self.performForwardDFT(timeDomainValues: &self.timeDomainValues,
                                   frequencyDomainValues: &self.frequencyDomainBuffer,
                                   temporaryRealBuffer: &self.realParts,
                                   temporaryImaginaryBuffer: &self.imaginaryParts)
            
            vDSP.absolute(frequencyDomainBuffer, result: &frequencyDomainBuffer)
            print("Our Frequencey Domain buffer is now", frequencyDomainBuffer.count)
           
            var complexMatrix = [DSPComplex](repeating: DSPComplex(real: 0, imag: 0), count: 200)

            // Set up the split complex vector
            let realCount = self.realParts.count
            self.realParts.withUnsafeMutableBufferPointer { realPtr in
                self.imaginaryParts.withUnsafeMutableBufferPointer { imagPtr in
                    var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                       imagp: imagPtr.baseAddress!)

                    complexMatrix.withUnsafeMutableBytes { complexMatrixUnsafe in
                        vDSP_ztoc(&splitComplex, 1,
                                  complexMatrixUnsafe.bindMemory(to: DSPComplex.self).baseAddress!, 2,
                                  vDSP_Length(realCount))
                    }
                }
            }
            
//            vDSP.absolute(complexMatrix, result: &complexMatrix)
            print("Our complex buffer is now", complexMatrix.count)

        }
        
        
        
        
        self.frequencyDomainBuffer.withUnsafeBufferPointer { frequencyDomainValuesPtr in
            
            self.melFilterBank.withUnsafeBufferPointer { melFilterBankValuesPtr in

                // cache the result of our count prior
                let sgemmResultCount = sgemmResult.count;
                self.sgemmResult.withUnsafeMutableBufferPointer { sgemmResultValuesPtr in
                    
                    // Multiplies our filter bank and our frequencyDomainBuffer
                    cblas_sgemm(CblasRowMajor,
                                CblasTrans, CblasTrans,
                                Int32(1),
                                Int32(self.melFilterBankCount),
                                Int32(self.melSampleCount),
                                1,
                                frequencyDomainValuesPtr.baseAddress, Int32(1),
                                melFilterBankValuesPtr.baseAddress, Int32(self.melSampleCount),
                                0,
                                sgemmResultValuesPtr.baseAddress, Int32(self.melFilterBankCount))
                    
                    // Converts single-precision power or amplitude values to decibel values.
                    vDSP_vdbcon(sgemmResultValuesPtr.baseAddress!, 1,
                                [20_000],
                                sgemmResultValuesPtr.baseAddress!, 1,
                                vDSP_Length(sgemmResultCount),
                                0)
                }
            }
        }
       
//        melSpectrumValues.append(contentsOf: sgemmResult)
        
    }
    
  
    
    /// Performs a forward Fourier transform on interleaved `timeDomainValues` writing the result to
    /// interleaved `frequencyDomainValues`.
    func performForwardDFT(timeDomainValues: inout [Float],
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
                              vDSP_Length(self.numFFT / 2))
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
                        
                        let log2n = vDSP_Length(log2(Float(self.numFFT)))
                        
                        vDSP_fft_zript(self.fft,
                                       &splitComplex, 1,
                                       &bufferSplitComplex,
                                       log2n,
                                       FFTDirection(kFFTDirection_Forward))
                    }
                }
            }
        }
        
//        // Populate interleaved `frequencyDomainValues` with the split values
//        // from the real and imaginary arrays.
//        temporaryRealBuffer.withUnsafeMutableBufferPointer { realPtr in
//            temporaryImaginaryBuffer.withUnsafeMutableBufferPointer { imagPtr in
//                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
//                                                   imagp: imagPtr.baseAddress!)
//
//                frequencyDomainValues.withUnsafeMutableBytes { ptr in
//                    vDSP_ztoc(&splitComplex, 1,
//                              ptr.bindMemory(to: DSPComplex.self).baseAddress!, 2,
//                              vDSP_Length(self.numFFT / 2))
//                }
//            }
//        }
    }

    func generateSpectrogram(audio: [Float]) -> [[Float]] {
        
        processData(audio: audio)
        
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

