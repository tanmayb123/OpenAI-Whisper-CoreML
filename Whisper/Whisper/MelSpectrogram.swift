//
//  stft.swift
//  Whisper
//
//  Created by Tanmay Bakshi on 2022-09-26.
//
import Accelerate

// Reference implementation we are attempting to match
// https://github.com/openai/whisper/blob/main/whisper/audio.py#L92

// See
// https://colab.research.google.com/drive/1r9ghakH8__jGqGiYHC2DXtKaW_ozdSrV?usp=sharing
// For simple isolated code to test this implementation

/*
    window = torch.hann_window(N_FFT).to(audio.device)
    1, 2, 3 - stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    4 - magnitudes = stft[:, :-1].abs() ** 2

    5 - filters = mel_filters(audio.device, n_mels)
    6 - mel_spec = filters @ magnitudes

    7 - log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    8 - log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    9 - log_spec = (log_spec + 4.0) / 4.0

    // Reference shapes - note - we dont match perfectly
    stft torch.Size([201, 3001])
    magnitudes torch.Size([201, 3000])
    mel filters torch.Size([80, 201])
    mel spec torch.Size([80, 3000])
    log spec torch.Size([80, 3000])
 
 */

// https://pytorch.org/docs/stable/generated/torch.stft.html
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/SpectralOps.cpp#L820
// SEE https://dsp.stackexchange.com/questions/49184/stft-amplitude-normalization-librosa-library
// See https://github.com/Jounce/Surge/issues/94
// see https://github.com/abokhalel2/istft/blob/main/swift/istft/ViewController.swift

// Some notes
// We do not calculate a 3001 mel, we skip the last since it wont be used anyway and is dropped later, saving 1/3000th of work.
//


// alternatively
// http://www.ml-illustrated.com/2020/06/01/deploy-pytorch-model-with-coreml-convert-issues.html
// and https://github.com/ml-illustrated/Pytorch-CoreML-Spectrogram/blob/d0dd6c55eaf5fdcfaf00b1f036b258bd144b1ac4/python/model.py#L142

public class MelSpectrogram
{
    // MARK: Properties
     
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

        /// The forward fast Fourier transform object.
    var fft: ComplexFFTStandard


    init(sampleCount:Int, hopCount:Int, melCount:Int, numFFT:Int)
    {
        self.sampleCount = sampleCount
        self.hopCount = hopCount
        self.melFilterBankCount = melCount
        
        self.melSampleCount = self.sampleCount / self.hopCount

        
        self.melFilterMatrix = MelSpectrogram.makeFilterBankWithNumpyData()
        
        self.fft = ComplexFFTStandard(numFFT: numFFT)
    }

    func processData(audio: [Int16]) -> [Float]
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
        for (m) in 0 ..< self.melSampleCount
        {
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            // audioFrame ends up holding split complex numbers
            
            // TODO: Handle Pytorch STFT Defaults:
            // TODO: Handle Centering  = True
            // TODO: Handle Padding = Reflect
            let audioFrame = Array<Float>( audioFloat[ (m * self.hopCount) ..< ( (m * self.hopCount) + self.fft.numFFT ) ] )

            assert(audioFrame.count == self.fft.numFFT)

            let (real, imaginary) = self.fft.forward(audioFrame)
            
            assert(real.count == self.fft.numFFT/2)
            
            allSampleReal.append(real)
            allSampleImaginary.append(imaginary)
        }
        
        // We now have allSample Split Complex holding 3000  200 dimensional real and imaginary FFT results
        // We create flattened  3000 x 200 array of DSPSplitComplex values
        var flattnedReal:[Float] = allSampleReal.flatMap { $0 }
        var flattnedImaginary:[Float] = allSampleImaginary.flatMap { $0 }

        print("Swift 0 - complex real min", vDSP.minimum(flattnedReal), "max", vDSP.maximum(flattnedReal))
        print("Swift 0 - complex imag min", vDSP.minimum(flattnedImaginary), "max", vDSP.maximum(flattnedImaginary))
        
        // Take the magnitude squared of the matrix, which results in a Result flat array of 3000 x 200 of real floats
        // Then multiply it with our mel filter bank
        var magnitudes = [Float](repeating: 0, count: flattnedReal.count)
        var melSpectroGram = [Float](repeating: 0, count: 80 * 3000)

        flattnedReal.withUnsafeMutableBytes { unsafeFlatReal in
            flattnedImaginary.withUnsafeMutableBytes { unsafeFlatImaginary in
                
                // We create a Split Complex representation of our flattened real and imaginary component
                let complexMatrix = DSPSplitComplex(realp: unsafeFlatReal.bindMemory(to: Float.self).baseAddress!,
                                                    imagp: unsafeFlatImaginary.bindMemory(to: Float.self).baseAddress!)

                vDSP.squareMagnitudes(complexMatrix, result: &magnitudes)
        
                print("Swift 1 - magnitudes min", vDSP.minimum(magnitudes), "max", vDSP.maximum(magnitudes))
                
                // transpose magnitudes from 3000 X 200, to 200 x 3000
                vDSP_mtrans(magnitudes, 1, &magnitudes, 1, 200, 3000) // verified correct

                // Step 5 & 6 (filters loaded earlier)

                // MATRIX A, a MxK sized matrix
                // MATRIX B, a KxN sized matrix
                // MATRIX C, a MxN sized matrix

                // MATRIX A mel filters is 80 rows x 200 columns
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
                            CblasNoTrans,           //
                            M,                      // M Number of rows in matrices A and C.
                            N,                      // N Number of columns in matrices B and C.
                            K,                      // K Number of columns in matrix A; number of rows in matrix B.
                            1.0,                    // Alpha Scaling factor for the product of matrices A and B.
                            self.melFilterMatrix,   // Matrix A
                            K,                      // LDA The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
                            magnitudes,             // Matrix B
                            N,                      // LDB The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                            0,                      // Beta Scaling factor for matrix C.
                            &melSpectroGram,        // Matrix C
                            N)                      // LDC The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                
                
                print("Swift 2 - mel min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))
                
                // Step 7 - clamp / clip the min to 1e-10
                vDSP.threshold(melSpectroGram, to: 1e-10, with: .clampToThreshold, result: &melSpectroGram)

                print("Swift 3 - mel clip min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))
                
                // Step 7 - Take the log base 10 - vDSP_vdbcon and power:toDecibels seems to fuck things up here and isnt right, even though its what everyone else uses?
                vForce.log10(melSpectroGram, result: &melSpectroGram)
                print("Swift 4 - mel log min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))
              
                // Step 8 -
                // Clip to new max and updated min
                let newMin = vDSP.maximum(melSpectroGram) - 8.0
                vDSP.maximum(melSpectroGram, [Float](repeating: newMin, count: melSpectroGram.count), result: &melSpectroGram)
            
                print("Swift 5 - mel log min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

                // Step 9 - Add 4 and Divide by 4
                vDSP.add(4.0, melSpectroGram, result: &melSpectroGram)
                vDSP.divide(melSpectroGram, 4.0, result: &melSpectroGram)
                print("Swift 6 - mel log norm min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

                print("--------------")

                print("Torch 0 - complex real min -11.8792142868) max 12.0689258575")
                print("Torch 0 - complex imag min -10.5751876831) max 11.5213479996")
                print("Torch 1 - magnitudes min 0.0000000000 max 165.6671142578")
                print("Torch 2 - mel min 0.0000000036 max tensor(4.2800636292)")
                print("Torch 3 - mel clip min 0.0000000036 max 4.2800636292")
                print("Torch 4 - mel log min -8.4495277405 max 0.6314502358")
                print("Torch 5 - mel log min -7.3685498238 max 0.6314502358")
                print("Torch 6 - mel log norm min -0.8421374559 max 1.1578625441")
            }
        }
        
        return melSpectroGram
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
    
    static func loadReferencePythonRawMelToDebugShit() -> [Float] {
    //        let numpyFloatArrayLength = 16080
        let fileURL = Bundle.main.url(forResource: "python_log_mel", withExtension:"raw")
        let fileHandle = try! FileHandle(forReadingFrom: fileURL!)

        let floatData = fileHandle.readDataToEndOfFile()
        let floatArray = floatData.withUnsafeBytes { unsafeFloatArray in
            return Array(UnsafeBufferPointer<Float>(start: unsafeFloatArray.bindMemory(to: Float.self).baseAddress!, count: floatData.count / MemoryLayout<Float>.stride /*(80 * 3000)*/) )
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
