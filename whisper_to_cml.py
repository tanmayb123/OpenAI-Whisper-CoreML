import whisper
import numpy as np
import torch
import coremltools as ct

def load_models():
    model = whisper.load_model("small").cpu()
    return model.encoder, model.decoder

def convert_encoder_to_tvm(model):
    model.eval()

    input_shape = (1, 80, 3000)
    input_data = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, input_data)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="logmel_data", shape=input_shape)]
    )

    return model

def convert_decoder_to_tvm(model):
    model.eval()

    tokens_shape = (1, 1)
    audio_shape = (1, 1500, 768)
    token_data = torch.randn(tokens_shape).long()
    audio_data = torch.randn(audio_shape)
    traced_model = torch.jit.trace(model, (token_data, audio_data))

    token_flexible_shape = ct.Shape(shape=(1,
                              ct.RangeDim(lower_bound=1, upper_bound=-1, default=1)))


    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_data", shape=token_flexible_shape, dtype=int),
            ct.TensorType(name="audio_data", shape=audio_shape)
        ]
    )

    return model

def main():
    encoder, decoder = load_models()

    decoder = convert_decoder_to_tvm(decoder)
    decoder.save("decoder.mlpackage")

    encoder = convert_encoder_to_tvm(encoder)
    encoder.save("encoder.mlpackage")

if __name__ == "__main__":
    main()
