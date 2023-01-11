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

    var numFFT:Int = 400

    /// The forward fast Fourier transform object.
    var fft: vDSP.FFT<DSPSplitComplex>

    /// The window sequence used to reduce spectral leakage.
    var hanningWindow:[Float]

    init(sampleCount:Int, hopCount:Int, melCount:Int, numFFT:Int)
    {
        self.sampleCount = sampleCount
        self.hopCount = hopCount
        self.melFilterBankCount = melCount
        
        self.melSampleCount = self.sampleCount / self.hopCount
        
        self.numFFT = numFFT
                
        self.hanningWindow = vDSP.window(ofType: Float.self,
                                         usingSequence: .hanningNormalized,
                                         count: self.numFFT,
                                         isHalfWindow: false)
        
        let log2n = vDSP_Length(log2(Float(self.numFFT)))

        self.fft = vDSP.FFT(log2n: log2n,
                           radix: .radix2,
                           ofType: DSPSplitComplex.self)!

        self.melFilterMatrix = MelSpectrogram.makeFilterBankWithNumpyData()
    }
    
    func processData(audio: [Int16]) -> [Float]
    {
        // Step 1

        // Audio has possible ranges of Signed Int - matches Pytorch Exactly
        //0, 0, 0, -1, 5, -27, -70, -58, -35, -21, -21, -23, -54, -59, -40, -56, -57, -15, -20, -8, 13, 20, 65, 46, 19, 51, 73, 71, 65, 79, 93, 93, 87, 84, 111, 92, 94, 152, 185, 189, 194, 194, 195, 184, 235, 271, 266, 250, 281, 294, 260, 266, 257, 234, 266, 268, 224, 212, 201, 196, 214, 207, 199, 186, 175, 174, 139, 81, 95, 128, 110, 111, 115, 101, 103, 108, 114, 96, 77, 77, 62, 29, 37, 51, 58, 90, 91, 63, 84, 81, 64, 77, 66, 31, 33, 22, -32, -42, -41, -51, -62
        assert(self.sampleCount == audio.count)
            
        var audioFloat:[Float] = [Float](repeating: 0, count: audio.count)
                
        vDSP.convertElements(of: audio, to: &audioFloat)
        // Audio now in Float, at Signed Int ranges - matches Pytorch Exactly


        vDSP.divide(audioFloat, 32768.0, result: &audioFloat)
        
        // Audio now in -1.0 to 1.0 Float ranges - matches Pytorch exactly

        // insert numFFT/2 samples before and numFFT/2 after so we have a extra numFFT amount to process
        // TODO: Is this stricly necessary?
        audioFloat.insert(contentsOf: [Float](repeating: 0, count: self.numFFT/2), at: 0)
        audioFloat.append(contentsOf: [Float](repeating: 0, count: self.numFFT/2))

//        audioFloat.append(contentsOf: [Float](repeating: 0, count: self.numFFT))

        // Split Complex arrays holding the FFT results
        var allSampleReal = [[Float]](repeating: [Float](repeating: 0, count: self.numFFT/2), count: self.melSampleCount)
        var allSampleImaginary = [[Float]](repeating: [Float](repeating: 0, count: self.numFFT/2), count: self.melSampleCount)

        // Step 2 - we need to create 3000 x 200 matrix of windowed FFTs
        // Pytorch outputs complex numbers
        for (m) in 0 ..< self.melSampleCount
        {
            // Slice numFFTs every hop count (barf) and make a mel spectrum out of it
            // audioFrame ends up holding split complex numbers
            var audioFrame = Array<Float>( audioFloat[ (m * self.hopCount) ..< ( (m * self.hopCount) + self.numFFT) ] )
            
            assert(audioFrame.count == self.numFFT)
            
//            for (k) in 0 ..< self.numFFT
//            {
//                hanningWindow[k] * audioFrame[ m * self.hopCount + k]
//            }

            
            // Split Complex arrays holding a single FFT result of our Audio Frame, which gets appended to the allSample Split Complex arrays
            var sampleReal:[Float] = [Float](repeating: 0, count: self.numFFT/2)
            var sampleImaginary:[Float] = [Float](repeating: 0, count: self.numFFT/2)


            sampleReal.withUnsafeMutableBytes { unsafeReal in
                sampleImaginary.withUnsafeMutableBytes { unsafeImaginary in

                    vDSP.multiply(audioFrame,
                                  hanningWindow,
                                  result: &audioFrame)

                    var complexSignal = DSPSplitComplex(realp: unsafeReal.bindMemory(to: Float.self).baseAddress!,
                                                        imagp: unsafeImaginary.bindMemory(to: Float.self).baseAddress!)

                    audioFrame.withUnsafeBytes { unsafeAudioBytes in
                        vDSP.convert(interleavedComplexVector: [DSPComplex](unsafeAudioBytes.bindMemory(to: DSPComplex.self)),
                                     toSplitComplexVector: &complexSignal)
                    }

                    // Step 3 - creating the FFT
                    self.fft.forward(input: complexSignal, output: &complexSignal)


                }
            }

            allSampleReal[m] = sampleReal
            allSampleImaginary[m] = sampleImaginary
        }
        
        // We now have allSample Split Complex holding 3000  200 dimensional real and imaginary FFT results
        
        // We create flattened  3000 x 200 array of DSPSplitComplex values
        var flattnedReal:[Float] = allSampleReal.flatMap { $0 }
        var flattnedImaginary:[Float] = allSampleImaginary.flatMap { $0 }

        // Take the magnitude squared of the matrix, which results in a Result flat array of 3000 x 200 of real floats
        // Then multiply it with our mel filter bank
        let count = flattnedReal.count
        var magnitudes = [Float](repeating: 0, count: count)
        var melSpectroGram = [Float](repeating: 0, count: 80 * 3000)
        
        flattnedReal.withUnsafeMutableBytes { unsafeFlatReal in
            flattnedImaginary.withUnsafeMutableBytes { unsafeFlatImaginary in
                
                // We create a Split Complex representation of our flattened real and imaginary component
                let complexMatrix = DSPSplitComplex(realp: unsafeFlatReal.bindMemory(to: Float.self).baseAddress!,
                                                    imagp: unsafeFlatImaginary.bindMemory(to: Float.self).baseAddress!)
                                    
                
                // Complex Matrix now has values like
                // (0.268574476 (real), -0.00511540473 (img))
                //
                // Important - In our Python notebook linked at the top
                // the STFT[0] and STFT[200] both have ZERO imaginary components
                // ONLY STFT[1] through STFT[199]
                //
                // print(stft[0][0])
                // tensor(0.34478074+0.j)
                //
                // print(stft[1][0])
                // tensor(-0.34575820-0.00000001j)
                
                // Python: print(stft[0][0:10])
                // tensor([ 0.34478074+0.j,  0.20181805+0.j,  0.36440092+0.j, -0.18822582+0.j, 0.82723594+0.j, -0.91735524+0.j, -0.10961355+0.j,  0.70801342+0.j, -0.74703455+0.j,  0.01316542+0.j])
                
                // Step 4 -
                // populate magnitude matrix with magnitudes squared
                // Magnitudes now contains single float 32
//                vDSP_zvmags(complexMatrix, 1, &magnitudes, 1, vDSP_Length(count))
                vDSP.squareMagnitudes(complexMatrix, result: &magnitudes)

                // Magitude Values
                // 0.07215842, 0.07432404, 0.07712047, 0.07292664,
                // 0.057917558, 0.036187824, 0.01670678, 0.006245121,
                // 0.0048390464, 0.007295049
                // Python: print(magnitudes[0][0:10])
                // tensor([    0.11887376,     0.04073052,     0.13278803,     0.03542896,
                //             0.68431932,     0.84154063,     0.01201513,     0.50128299,
                //             0.55806065,     0.00017333])
                
                // Similar range, but very different values
                
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
                
                // Step 7 - clamp / clip the min to 1e-10
//                vDSP.clip(melSpectroGram, to: (1e-10)...(vDSP.maximum(melSpectroGram)), result: &melSpectroGram)
                
                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))
                
                vDSP.threshold(melSpectroGram, to: 1e-10, with: .clampToThreshold, result: &melSpectroGram)

                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

                // Step 7 - Take the log base 10
                // vDSP_vdbcon and power:toDecibels seems to fuck things up here and isnt right, even though its what everyone else uses?
                vForce.log10(melSpectroGram, result: &melSpectroGram)

                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))
                // Step 8 -
                // Clip to new max and updated min
                let newMin = vDSP.maximum(melSpectroGram) - 8.0
//                vDSP.clip(melSpectroGram, to: (newMin)...(vDSP.maximum(melSpectroGram)), result: &melSpectroGram)

//                vDSP.maximum(melSpectroGram, [Float](repeating: newMin, count: melSpectroGram.count), result: &melSpectroGram)
  
                vDSP.limit(melSpectroGram, limit:newMin, withOutputConstant: newMin, result:&melSpectroGram)
                
                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

                // Step 9 - Add 4 and Divide by 4
                vDSP.add(4.0, melSpectroGram, result: &melSpectroGram)
                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

                vDSP.divide(melSpectroGram, 4.0, result: &melSpectroGram)
                print("min", vDSP.minimum(melSpectroGram), "max", vDSP.maximum(melSpectroGram))

            }
        }
        // At this point we have a Log Mel Spectrogram who has values between -4 and 4
        // Samples values:
        // 0.31666893, 0.48593897, 0.4478705, 0.55086285, 0.7317668, 0.6822369, 0.60241175, 0.5754051, 0.5703093, 0.5082297
        // Python Values:
        // print(log_spec[0][0:10])
        // tensor([ 0.36827284, 0.29459417, 0.33011752, 0.40674573, 0.58850896, 0.60998225,
        //          0.57963496, 0.53520113, 0.51033145, 0.42032439])
        
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
