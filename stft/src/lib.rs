use std::f64::consts::PI;
use std::include_bytes;
use std::sync::Arc;
use num::complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use lazy_static::lazy_static;
use npy::NpyData;

const NPY_BYTES: &'static [u8] = include_bytes!("m80.npy");

lazy_static! {
    static ref GENERATOR: Arc<SpectrogramGenerator> = Arc::new(SpectrogramGenerator::new());
    static ref MELS: Vec<f32> = load_npy_bytes(NPY_BYTES);
}

struct SpectrogramGenerator {
    fft_plan: Arc<dyn RealToComplex<f64>>,
    window: Vec<f64>
}

impl SpectrogramGenerator {
    fn new() -> SpectrogramGenerator {
        let mut planner = RealFftPlanner::new();
        let fft_plan = planner.plan_fft_forward(400);

        let window = ((0..400).map(|i| (i as f64 * 2.0 * PI) / 400.0).map(|i| (1.0 - i.cos()) / 2.0).collect::<Vec<_>>()).to_vec();

        SpectrogramGenerator {
            fft_plan,
            window,
        }
    }

    fn reflect(audio: &mut [f64]) {
        for i in 0..200 {
            audio[i] = audio[400 - i];
            let j = 16000 * 30 + i + 200;
            audio[j] = audio[200 + (16000 * 30 - 2) - i];
        }
    }

    fn fft(&self, audio: &[f64]) -> Vec<Complex<f64>> {
        let mut input = audio.iter().zip(self.window.iter()).map(|(a, w)| a * w).collect::<Vec<_>>();
        let mut spectrum = self.fft_plan.make_output_vec();
        self.fft_plan.process(&mut input, &mut spectrum).unwrap();
        spectrum
    }

    fn spectrogram(&self, audio: &[f64]) -> Vec<Vec<f64>> {
        let stride_size = 160;
        let mut spectrogram = (0..201).map(|_| Vec::new()).collect::<Vec<_>>();
        for i in (0..audio.len() - 400).step_by(stride_size) {
            let fft = self.fft(&audio[i..i+400]);
            let spectrogram_row = fft.iter().map(|c| c.norm_sqr()).collect::<Vec<_>>();
            for i in 0..201 {
                spectrogram[i].push(spectrogram_row[i]);
            }
        }

        let mut processed = (0..80).map(|_| Vec::new()).collect::<Vec<_>>();
        for i in 0..80 {
            for j in 0..3000 {
                let mut sum = 0.0;
                for k in 0..201 {
                    sum += spectrogram[k][j] * MELS[i * 201 + k] as f64;
                }
                processed[i].push(sum);
            }
        }

        processed = processed
            .into_iter()
            .map(|row| {
                row
                    .into_iter()
                    .map(|x| x.max(1e-10).log10())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();


        let processed_max = *{
            &processed
                .iter()
                .map(|row| *(row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()))
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        };


        processed = processed
            .into_iter()
            .map(|row| {
                row
                    .into_iter()
                    .map(|x| (x.max(processed_max - 8.0) + 4.0) / 4.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        processed
    }
}

fn load_npy_bytes(bytes: &[u8]) -> Vec<f32> {
    let data: NpyData<f32> = NpyData::from_bytes(bytes).unwrap();
    data.to_vec().iter().map(|x| *x as f32).collect()
}

#[no_mangle]
pub extern fn generate_spectrogram(audio: *mut f64, output: *mut f64) {
    let mut audio = unsafe { std::slice::from_raw_parts_mut(audio, 16000 * 30 + 400) };
    SpectrogramGenerator::reflect(&mut audio);
    let spectrogram = GENERATOR.spectrogram(audio);

    let output = unsafe { std::slice::from_raw_parts_mut(output, 80 * 3000) };
    for i in 0..80 {
        for j in 0..3000 {
            output[i * 3000 + j] = spectrogram[i][j];
        }
    }
}
