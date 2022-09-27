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
        inputs=[ct.TensorType(shape=input_shape)]
    )

    return model

def convert_decoder_to_tvm(model):
    model.eval()

    tokens_shape = (1, 1)
    audio_shape = (1, 1500, 768)
    token_data = torch.randn(tokens_shape).long()
    audio_data = torch.randn(audio_shape)
    traced_model = torch.jit.trace(model, (token_data, audio_data))

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(shape=tokens_shape),
            ct.TensorType(shape=audio_shape)
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
