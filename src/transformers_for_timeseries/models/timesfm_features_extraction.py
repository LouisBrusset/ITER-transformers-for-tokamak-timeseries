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
    
    def build_weighted_matrix(self, features, threshold: float = 0.001):
        """
        We realize that some features extracted by TimesFm are really similar, particularly when the input sequence is short.
        In details, for short input sequences, the n first patches are equal to each other, and the information stays only in the N-n last patches.
        So realizing the mean over all patches features break the flow of information.
        To avoid this, we will yield only the number of useful components in the feature sequence.
        After several experiments, we define useful components as those whose RMSE with respect to the first feature vector is above a certain threshold. (RMSE with the mean vector didn't give good results).
        Then the number of useful components is the number of feature vectors satisfying this condition, counting from the last one (due to TimesFM implementation).

        Args:
            features (torch.Tensor): Tensor of shape (batch_size, n_patches, feature_size)
            threshold (float): Threshold to determine useful components.

        Returns:
            torch.Tensor: shape (batch_size, number_of_patches) 
                          Matrix of weights (0 or 1) indicating useful components for each sample in the batch.
        """
        rmse = torch.sqrt(torch.mean((features - features[:, 0, :][:, None, :]) ** 2, dim=2))  # shape (batch_size, n_patches)
        useful_components = torch.sum(rmse > threshold, axis=1) # shape (batch_size,)

        matrix_weights = torch.zeros_like(features[:, :, 0])  # shape (batch_size, n_patches)
        for i in range(features.shape[0]):
            n_useful = useful_components[i]
            if n_useful > 0:
                matrix_weights[i, -n_useful:] = 1.0

        return matrix_weights