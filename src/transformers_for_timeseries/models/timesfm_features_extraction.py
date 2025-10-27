import torch
from transformers import TimesFmModelForPrediction

class TimesFmFeatureExtractor:
    def __init__(self, model_name="google/timesfm-2.0-500m-pytorch"):
        self.model = TimesFmModelForPrediction.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        
        self.intercepted_features = {}
        self.hook_handle = self._register_hook()
    
    def _register_hook(self):
        """Place the hook on the decoder to intercept its output"""
        
        def decoder_hook(module, input, output):
            """
            This function will be called automatically after the decoder's forward pass
                module: the TimesFmDecoder module
                input: tuple containing the decoder's inputs
                output: the decoder's outputs (what we want to capture)
            """
            self.intercepted_features['decoder_output'] = output["last_hidden_state"].detach().clone()
            return output       # return the output unchanged not to disrupt the forward pass

        hook_handle = self.model.decoder.register_forward_hook(decoder_hook)
        return hook_handle
    
    def extract_features(self, past_values, freq):
        """
        Forward pass through the model to extract features from the decoder.
            past_values: Tensor of shape (batch_size, seq_len)
            freq: Tensor of shape (batch_size,)
        """
        self.intercepted_features.clear()
        
        with torch.no_grad():
            _ = self.model(
                past_values=past_values,
                freq=freq
            )
            if 'decoder_output' in self.intercepted_features:
                features = self.intercepted_features['decoder_output']
                return features
            else:
                raise RuntimeError("No features intercepted by the hook")
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()