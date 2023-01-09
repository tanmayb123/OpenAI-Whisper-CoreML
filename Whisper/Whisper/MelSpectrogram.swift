//
//  stft.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//
import Accelerate

// Reference implementation we are attempting to match
// https://github.com/openai/whisper/blob/main/whisper/audio.py#L92

// See https://colab.research.google.com/drive/1r9ghakH8__jGqGiYHC2DXtKaW_ozdSrV#scrollTo=7baNvPkScRgk
// For simple isolated code to test this implementation

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

// Some notes
// We do not calculate a 3001 mel, we skip the last since it wont be used anyway and is dropped later, saving 1/3000th of work.
//


// alternatively
// http://www.ml-illustrated.com/2020/06/01/deploy-pytorch-model-with-coreml-convert-issues.html
// and https://github.com/ml-illustrated/Pytorch-CoreML-Spectrogram/blob/d0dd6c55eaf5fdcfaf00b1f036b258bd144b1ac4/python/model.py#L142

public class MelSpectrogram
{
    // MARK: Properties
    
    /// An 2D  array that contains the entire spectrogram, 80 x 3000
    var melSpectrumValues:[[Float]]
    
    /// a 3000 x 201 array - we compute 201 complex numbers for each STFT we compute
    /// this is transposed to 201 x 3000 and matix multiplies by our 80 x 201 mel filter matrix
    /// in order to optimize this, we split our array into real and imaginary components so we can  more easily use our
    var complexSTFTReal:[[Float]]
    var complexSTFTImaginary:[[Float]]
 
    /// windows for each mel frequency
    /// Our 80 x 201 sized matrix of 16080 float values of precomputed filters.
    var melFilterMatrix:[Float]
    
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
        
        self.melFilterMatrix = MelSpectrogram.makeFilterBankWithNumpyData()
        
        self.complexSTFTReal = [[Float]](repeating: [Float](repeating: 0, count: self.numFFT/2), count: self.melSampleCount)
        self.complexSTFTImaginary = [[Float]](repeating: [Float](repeating: 0, count: self.numFFT/2), count: self.melSampleCount)
    }
    
    func processData(audio: [Int16]) -> [Float]
    {
        assert(self.sampleCount == audio.count)
            
        var audioFloat:[Float] = [Float](repeating: 0, count: audio.count)
        vDSP.convertElements(of: audio, to: &audioFloat)
              
        vDSP.divide(audioFloat, 32768.0, result: &audioFloat)
        
        // insert numFFT/2 samples before and numFFT/2 after so we have a extra numFFT amount to process
        audioFloat.insert(contentsOf: [Float](repeating: 0, count: self.numFFT/2), at: 0)
        audioFloat.append(contentsOf: [Float](repeating: 0, count: self.numFFT/2))
        
        // we need to create 201 x 3000 matrix of STFTs - note we appear to want to output complex numbers (?)
        for (i) in 0 ..< self.melSampleCount
        {
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            self.timeDomainValues = Array<Float>( audioFloat[ (i * self.hopCount) ..< ( (i * self.hopCount) + self.numFFT) ] )
            
            assert(self.timeDomainValues.count == self.numFFT)
            
            self.performForwardDFT(timeDomainValues: &self.timeDomainValues,
                                   frequencyDomainValues: &self.frequencyDomainBuffer,
                                   temporaryRealBuffer: &self.realParts,
                                   temporaryImaginaryBuffer: &self.imaginaryParts)
            
//            vDSP.absolute(frequencyDomainBuffer, result: &frequencyDomainBuffer)
//            vDSP.absolute(self.realParts, result: &self.realParts)
//            vDSP.absolute(self.imaginaryParts, result: &self.imaginaryParts)

            self.complexSTFTReal[i] = self.realParts
            self.complexSTFTImaginary[i] = self.imaginaryParts
        }
        
        // We create flattened  3000 x 200 array of DSPSplitComplex values
        var flattnedReal:[Float] = self.complexSTFTReal.flatMap { $0 }
        var flattnedImaginary:[Float] = self.complexSTFTImaginary.flatMap { $0 }

        
        // Take the magnitude squared of the matrix, which results in a Result flat array of 3000 x 200 of real floats
        // Then multiply it with our mel filter bank
        let count = self.complexSTFTReal.count * self.complexSTFTReal[0].count
        var magnitudes = [Float](repeating: 0, count: count)
        var melSpectroGram = [Float](repeating: 0, count: count)

        flattnedReal.withUnsafeMutableBytes { unsafeReal in
            flattnedImaginary.withUnsafeMutableBytes { unsafeImaginary in

                let matrix = [DSPSplitComplex](repeating: DSPSplitComplex(realp: unsafeReal.bindMemory(to: Float.self).baseAddress!,
                                                                          imagp: unsafeImaginary.bindMemory(to: Float.self).baseAddress!),
                                               count: count)
               
               // populate magnitude matrix with magnitudes squared
               vDSP_zvmags(matrix, 1, &magnitudes, 1, vDSP_Length(count))
               
               // MATRIX A mel filters is 80 rows x 201 columns
               // MATRIX B magnitudes is 3000 x 200
               // MATRIX B is TRANSPOSED to be 200 rows x 3000 columns
               // MATRIX C melSpectroGram is 80 rows x 3000 columns
               
               let M: Int32 = 80 // number of rows in matrix A
               let N: Int32 = 3000 // number of columns in matrix B
               let K: Int32 = 200 // number of columns in matrix A and number of rows in
               
               // matrix multiply magitude squared matrix with our filter bank
               // see https://www.advancedswift.com/matrix-math/
               cblas_sgemm(CblasRowMajor,
                           CblasNoTrans,           // Transpose A
                           CblasTrans,             // Transpose B magnitudes to be 200 x 3000
                           M,                      // M Number of rows in matrices A and C.
                           N,                      // N Number of columns in matrices B and C.
                           K,                      // K Number of columns in matrix A; number of rows in matrix B.
                           1,                      // Alpha Scaling factor for the product of matrices A and B.
                           self.melFilterMatrix,   // Matrix A
                           K,                      // LDA The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
                           magnitudes,             // Matrix B
                           N,                      // LDB The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                           0,                      // Beta Scaling factor for matrix C.
                           &melSpectroGram,        // Matrix C
                           N)                      // LDC The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
               //        }
               
               var minValue: Float = 1e-10
               var maxValue: Float = 0.0
               var maxIndex: vDSP_Length = 0
               var minIndex: vDSP_Length = 0
               
               let melCount = melSpectroGram.count
               
               //        melSpectroGram.withUnsafeMutableBufferPointer { unsafeMelSpectrogram in
               
               //            let melBaseAddress = unsafeMelSpectrogram.baseAddress!
               
               // get the current max value
               vDSP_maxvi(melSpectroGram, 1, &maxValue, &maxIndex, vDSP_Length(melCount))
               
               // Clip to a set min value, keeping the current max value
               vDSP_vclip(melSpectroGram, 1, &minValue, &maxValue, &melSpectroGram, 1, vDSP_Length(melCount))
               
               // Take the log base 10
               var melCountInt32:UInt32 = UInt32(melCount)
               vvlog10f(&melSpectroGram, melSpectroGram, &melCountInt32)
               
               // get the new max value
               vDSP_maxvi(melSpectroGram, 1, &maxValue, &maxIndex, vDSP_Length(melCount))
               
               // get the new min value
               vDSP_minvi(melSpectroGram, 1, &minValue, &minIndex, vDSP_Length(melCount))
               
               // emulate
               // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
               // we effectively clamp to max - 8.0
               var newMin = maxValue - 8.0
               
               // Clip to new max and updated min
               vDSP_vclip(melSpectroGram, 1, &newMin, &maxValue, &melSpectroGram, 1, vDSP_Length(melCount))
               
               // Add 4 and Divide by 4
               
               var four:Float = 4.0
               vDSP_vsadd(melSpectroGram, 1, &four, &melSpectroGram, 1, vDSP_Length(melCount))
               vDSP_vsdiv(melSpectroGram, 1, &four, &melSpectroGram, 1, vDSP_Length(melCount))
               //        }
               
            }
        }
        
        return melSpectroGram

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
        
        // Populate interleaved `frequencyDomainValues` with the split values
        // from the real and imaginary arrays.
        temporaryRealBuffer.withUnsafeMutableBufferPointer { realPtr in
            temporaryImaginaryBuffer.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!,
                                                   imagp: imagPtr.baseAddress!)

                frequencyDomainValues.withUnsafeMutableBytes { ptr in
                    vDSP_ztoc(&splitComplex, 1,
                              ptr.bindMemory(to: DSPComplex.self).baseAddress!, 2,
                              vDSP_Length(self.numFFT / 2))
                }
            }
        }
    }


    static func makeFilterBankWithNumpyData() -> [Float] {
//        let numpyFloatArrayLength = 16080
        let fileURL = Bundle.main.url(forResource: "mel_filters", withExtension:"data")
        let fileHandle = try! FileHandle(forReadingFrom: fileURL!)

        let floatData = fileHandle.readDataToEndOfFile()
        let floatArray = floatData.withUnsafeBytes { unsafeFloatArray in
            return Array(UnsafeBufferPointer<Float>(start: unsafeFloatArray.bindMemory(to: Float.self).baseAddress!, count: floatData.count / MemoryLayout<Float>.stride) )
        }

        return floatArray;

    }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
