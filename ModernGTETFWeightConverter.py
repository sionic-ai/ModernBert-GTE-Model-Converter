from ModernGTETFModel import ModernGTETensorFlow, save_model_with_tokenizer

from transformers import AutoModel
import tensorflow as tf


class ModernGTEWeightConverter:
    def __init__(self, model_name_or_path: str):
        self.pt_model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.pt_state_dict = self.pt_model.state_dict()

        for name, param in self.pt_model.state_dict().items():
            print(f"{name:<30} | shape: {param.shape}")

    def initialize_weights(self, tf_model):
        """Initialize TensorFlow model with PyTorch weights"""
        # Build model with dummy inputs first
        dummy_inputs = {
            "input_ids": tf.zeros((2, 500), dtype=tf.int32),
            "attention_mask": tf.ones((2, 500), dtype=tf.int32),
        }
        _ = tf_model(dummy_inputs, training=False)

        # Initialize embeddings
        self._init_embedding_weights(tf_model)

        # Initialize encoder layers
        self._init_transformer_blocks(tf_model)

        # Initialize pooler
        self._init_pooler_weights(tf_model)

        return tf_model

    def _init_embedding_weights(self, tf_model):
        """Initialize embedding layer weights"""

        # Word embeddings
        tf_model.weight.assign(
            self.pt_state_dict["embeddings.tok_embeddings.weight"].numpy()
        )

        # Layer normalization
        tf_model.layerNorm.set_weights(
            [self.pt_state_dict["embeddings.norm.weight"].numpy()]
        )

    def _init_transformer_blocks(self, tf_model):
        """Initialize transformer block weights"""
        for i, layer in enumerate(tf_model.encoder_layers):
            prefix = f"layers.{i}."

            # Attention weights
            self._init_attention_weights(layer, prefix)

            # Feed-forward weights
            self._init_ffn_weights(layer, prefix)

    def _init_attention_weights(self, layer, prefix):
        """Initialize multi-head attention weights with correct reshaping"""
        # Load weights from PyTorch

        if prefix != "layers.0.":
            attn_lm_weight = self.pt_state_dict[f"{prefix}attn_norm.weight"].numpy()
            # Layer normalization
            layer.attention.attnNorm.set_weights([attn_lm_weight])

        qkv_weight = self.pt_state_dict[f"{prefix}attn.Wqkv.weight"].numpy()
        out_weight = self.pt_state_dict[f"{prefix}attn.Wo.weight"].numpy()

        # Set weights in the correct order for TensorFlow
        layer.attention.wqkv.set_weights(
            [
                qkv_weight.T,  # (num_heads, hidden_size, head_size)
            ]
        )

        layer.attention.o.set_weights([out_weight.T])

    def _init_ffn_weights(self, layer, prefix):
        """Initialize feed-forward network weights"""

        # Layer norm
        layer.mlp_norm.set_weights(
            [
                self.pt_state_dict[f"{prefix}mlp_norm.weight"].numpy(),
            ]
        )

        # Intermediate dense
        layer.Wi.set_weights([self.pt_state_dict[f"{prefix}mlp.Wi.weight"].numpy().T])

        # Output dense
        layer.Wo.set_weights([self.pt_state_dict[f"{prefix}mlp.Wo.weight"].numpy().T])

    def _init_pooler_weights(self, tf_model):
        """Initialize pooler weights"""

        final_norm_weight = self.pt_state_dict["final_norm.weight"].numpy()

        # 가중치 설정 전에 dummy forward pass로 build
        dummy_input = tf.random.normal([1, 10, final_norm_weight.shape[0]])
        _ = tf_model.final_norm(dummy_input[:, 0, :])  # [CLS] 토큰만 사용하도록 수정

        # 가중치 설정
        tf_model.final_norm.set_weights([final_norm_weight])


def convert_and_save_model(model_name: str, save_path: str):
    """Convert PyTorch model to TensorFlow and save"""
    # Initialize TensorFlow model
    tf_model = ModernGTETensorFlow(model_name)

    # Convert weights
    converter = ModernGTEWeightConverter(model_name)
    tf_model = converter.initialize_weights(tf_model)

    # Save model
    tokenizer = tf_model.tokenizer
    save_model_with_tokenizer(tf_model, tokenizer, save_path)

    return tf_model


if __name__ == "__main__":
    model_name = "Alibaba-NLP/gte-modernbert-base"
    save_path = "./converted_gte-modernbert-base"

    tf_model = convert_and_save_model(model_name, save_path)
    print("Model converted and saved successfully!")
