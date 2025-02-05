import os
import tensorflow as tf
from transformers import AutoTokenizer, AutoConfig
from typing import Optional

TFDTYPE = tf.float32


def create_local_sliding_window_mask(global_mask_4d, window_size):
    """
    ModernBERT의 local attention을 위한 "양방향 슬라이딩 윈도우" 마스크를 만든 뒤,
    원본 global_mask(패딩 토큰 마스킹)와 결합하여 최종 4D float 마스크를 반환합니다.

    - PyTorch 코드를 예시로:
        distance = |i - j|
        window_mask = (distance <= window_size//2)
        sliding_window_mask = global_attention_mask.masked_fill(~window_mask, -∞)
      에 대응.

    Args:
        global_mask_4d: shape (batch_size, 1, seq_len, seq_len)
            - 이미 패딩 토큰 부분은  -∞(float('-inf')) 또는 0.0 으로 구성된 float 마스크.
            - 보통 BERT식 마스크는 "유효 위치=0.0, 무효 위치=-∞" 형태임
        window_size (int): 로컬 윈도우 크기 (예: 128)

    Returns:
        final_local_4d (tf.Tensor):
            shape (batch_size, 1, seq_len, seq_len)
            - 윈도우 내부이면서 유효 토큰이면 0.0
            - 윈도우 밖이거나 패딩이면 -∞
    """
    # global_mask_4d: [B, 1, S, S]
    batch_size = tf.shape(global_mask_4d)[0]
    seq_len = tf.shape(global_mask_4d)[-1]

    # (S, S)에서의 distance 계산
    rows = tf.range(seq_len)[:, None]  # shape (S,1)
    cols = tf.range(seq_len)[None, :]  # shape (1,S)
    distance = tf.abs(rows - cols)  # shape (S,S)

    # distance가 window_size//2 이내면 True
    half_w = window_size // 2
    window_bool_2d = tf.less_equal(distance, half_w)  # (S,S), True/False

    # True면 0.0, False면 -∞인 2D float mask
    inside_0 = tf.zeros([seq_len, seq_len], dtype=TFDTYPE)
    outside_inf = tf.fill([seq_len, seq_len], TFDTYPE.min)  # -1e9)
    local_mask_2d = tf.where(window_bool_2d, inside_0, outside_inf)  # (S,S), float

    # shape 확장: [1,1,S,S]
    local_mask_4d = local_mask_2d[None, None, :, :]

    # batch 차원만큼 복제: [B,1,S,S]
    local_mask_4d = tf.tile(local_mask_4d, [batch_size, 1, 1, 1])

    # PyTorch의 sliding_window_mask = global_mask.masked_fill(~window_bool_2d, -inf)에 해당
    #   => local_mask_4d가 window 밖을 -∞로 만들어놓았으므로
    #      밖일 때 -∞, 안일 때 0.0
    #   => global_mask_4d도 이미 "패딩 부분 -∞, 정상 부분 0.0" 형태이므로
    #      합산하면 "둘 중 하나라도 -∞이면 -∞"라는 효과가 남

    final_local_4d = global_mask_4d + local_mask_4d  # (B,1,S,S)

    return final_local_4d


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1, layer_id=0, **kwargs):
        super().__init__(**kwargs)

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.layer_id = layer_id
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.head_dim = self.depth
        self.all_head_size = self.num_heads * self.head_dim

        self.attnNorm = tf.keras.layers.LayerNormalization(epsilon=1e-5, center=False)

        # QKV projection
        self.wqkv = tf.keras.layers.Dense(d_model * 3, use_bias=False)
        self.o = tf.keras.layers.Dense(d_model, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        # cos, sin shape이 [seq_len, 2*dim], 여기서 브로드캐스팅 위해 reshape
        cos = tf.reshape(cos, [1, 1, -1, self.head_dim])
        sin = tf.reshape(sin, [1, 1, -1, self.head_dim])

        def rotate_half(x):
            x1, x2 = tf.split(x, 2, axis=-1)
            return tf.concat([-x2, x1], axis=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def scaled_dot_product_attention(
        self,
        query,
        key,
        value,
        mask=None,  # 여기서 mask는 [batch_size, 1, seq_len, seq_len]
        training=None,
    ):
        """
        PyTorch처럼 'query @ key^T' 후 mask를 더해 Softmax
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)  # [..., seq_len, seq_len]

        dk = tf.cast(tf.shape(key)[-1], query.dtype)
        scaled_logits = matmul_qk / tf.math.sqrt(dk)

        # mask가 0.0 또는 -∞ 형태라고 가정하면, 그냥 더해주면 됨
        if mask is not None:
            scaled_logits = scaled_logits + mask  # -∞ 더해지는 곳 => softmax에서 0 으로

        # softmax
        attention_weights = tf.nn.softmax(scaled_logits, axis=-1)

        # dropout
        if training:
            attention_weights = tf.nn.dropout(attention_weights)

        output = tf.matmul(attention_weights, value)
        return output

    def __call__(self, inputs, mask=None, rope_embeds=None, training=False):
        """
        mask: 보통 [batch_size, 1, seq_len, seq_len] 형상
        rope_embeds: (cos, sin)
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        if self.layer_id != 0:
            inputs = self.attnNorm(inputs)

        # QKV
        qkv = self.wqkv(inputs)  # [B, S, 3*d_model]
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        # (3, B, head, S, head_dim)으로 transpose해도 되지만, 여기서는 아래처럼 unstack
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # [3, B, num_heads, S, head_dim]
        q, k, v = tf.unstack(qkv, axis=0)  # 각각 [B, num_heads, S, head_dim]

        # Rotary
        if rope_embeds is not None:
            cos, sin = rope_embeds
            q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        # SDPA (scaled_dot_product_attention)
        # shape: q,k => [B, heads, S, head_dim], v => [B, heads, S, head_dim]
        # mask => [B, 1, S, S], 브로드캐스팅을 위해 아래에서 reshape
        if mask is not None:
            # attention에서 head차원만큼 브로드캐스트. ex) mask를 [B, 1, S, S] -> [B, num_heads, S, S]
            # tf.tile 써도 되지만, 그냥 matmul에서 broadcasting 가능
            mask = tf.reshape(mask, [tf.shape(mask)[0], 1, seq_len, seq_len])

        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask=mask, training=training
        )
        # [B, heads, S, head_dim] -> [B, S, heads, head_dim]
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        # -> [B, S, d_model]
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model]
        )

        # output proj
        attention_output = self.o(attention_output)
        if training:
            attention_output = self.dropout(attention_output, training=training)

        return attention_output


class NTKScalingRotaryEmbedding(tf.keras.layers.Layer):
    """
    PyTorch의 ModernBertRotaryEmbedding(NTKScaling 등)과 대응.
    BERT이지만, local attention일 경우 max_position_embeddings를 줄여쓰거나,
    rope_theta를 다르게 쓸 수 있습니다.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 160000.0,
        scaling_factor: float = 1.0,
        mixed_b: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        max_position_embeddings = int(max_position_embeddings * scaling_factor)

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.mixed_b = mixed_b

        # 0,2,4,... 인덱스에 해당하는 inv_freq
        indices = tf.range(0, self.dim, 2, dtype=TFDTYPE)

        # 기본 base_inv_freq
        self.base_inv_freq = 1.0 / tf.pow(self.base, (indices / self.dim))

        # NTK 스케일링된 inv_freq
        if self.mixed_b is None:
            scaled_base = self.base * self.scaling_factor
            scaled_inv_freq = (1.0 / tf.pow(scaled_base, indices / self.dim)) / tf.pow(
                self.scaling_factor, 2.0 / self.dim
            )
        else:
            # 혼합 계수일 때 (여기선 생략 가능)
            scaled_base = self.base
            base_inv_freq = self.base_inv_freq
            a = tf.math.log(tf.cast(self.scaling_factor, TFDTYPE)) / tf.pow(
                self.dim / 2.0, self.mixed_b
            )
            indices_1_to_d2 = tf.range(1, self.dim // 2 + 1, dtype=TFDTYPE)
            lambda_1_m = tf.exp(a * tf.pow(indices_1_to_d2, self.mixed_b))
            scaled_inv_freq = base_inv_freq / lambda_1_m

        self.scaled_inv_freq = scaled_inv_freq

        # 초기 cache
        self._build_initial_cache()

    def _build_initial_cache(self):
        t = tf.range(self.max_position_embeddings, dtype=TFDTYPE)
        freqs = tf.einsum("i,j->ij", t, self.scaled_inv_freq)  # (max_len, dim/2)
        emb = tf.concat([freqs, freqs], axis=-1)  # (max_len, dim)
        self.cos_cached = tf.cos(emb)
        self.sin_cached = tf.sin(emb)

    def _compute_new_cache(self, seq_len):
        t = tf.range(seq_len, dtype=TFDTYPE)
        freqs = tf.einsum("i,j->ij", t, self.scaled_inv_freq)
        emb = tf.concat([freqs, freqs], axis=-1)
        return tf.cos(emb), tf.sin(emb)

    def call(self, x, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = tf.shape(x)[1]

        if tf.executing_eagerly():
            if seq_len > self.max_position_embeddings:
                cos, sin = self._compute_new_cache(seq_len)
            else:
                cos = self.cos_cached[:seq_len]
                sin = self.sin_cached[:seq_len]
        else:
            # 그래프 모드에선 tf.cond
            def use_new_cache():
                return self._compute_new_cache(seq_len)

            def use_cached():
                return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

            cos, sin = tf.cond(
                tf.greater(seq_len, self.max_position_embeddings),
                use_new_cache,
                use_cached,
            )

        cos = tf.cast(cos, x.dtype)
        sin = tf.cast(sin, x.dtype)
        return cos, sin


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        intermediate_size,
        dropout_rate=0.1,
        layer_id=0,
        config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = config
        self.layer_id = layer_id
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate, layer_id)

        # local / global 구분해서 rope_theta, max_position_embeddings 조정
        self.rope_theta = self.config.global_rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        if (self.layer_id % self.config.global_attn_every_n_layers) != 0:
            # local
            if self.config.local_rope_theta is not None:
                self.rope_theta = self.config.local_rope_theta
            self.max_position_embeddings = self.config.local_attention

        self.rotary_emb = NTKScalingRotaryEmbedding(
            dim=int(d_model // num_heads),
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            name=f"rotary_embeddings_{layer_id}",
        )

        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.mlp_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, center=False)

        self.Wi = tf.keras.layers.Dense(
            intermediate_size * 2, name="intermediate.dense", use_bias=False
        )
        self.Wo = tf.keras.layers.Dense(d_model, name="output.dense", use_bias=False)
        self.output_dropout = tf.keras.layers.Dropout(dropout_rate)

    def gelu_approx(self, x):
        # 원 ModernBertMLP에서 ACT2FN[config.hidden_activation]
        return tf.nn.gelu(x)

    def call(self, hidden_states, attention_mask=None, training=False):
        batch_size = tf.shape(hidden_states)[0]
        seq_len = tf.shape(hidden_states)[1]

        # RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
        rope_embeds = (cos, sin)

        # (1) local vs global
        #     local이면 window_size = config.local_attention
        #     global이면 None
        window_size = None
        if (self.layer_id % self.config.global_attn_every_n_layers) != 0:
            window_size = self.config.local_attention  # 예: 128

        # (2) global_mask_4d == attention_mask(기본 4D)
        #     만약 window_size가 있으면 local_sliding_window_mask로 변환
        if window_size is not None and window_size > 0:
            # attention_mask는 [batch, 1, seq_len, seq_len] 로 가정
            # local sliding window 마스크를 합성
            attention_mask = create_local_sliding_window_mask(
                attention_mask, window_size
            )
        # else => 그대로 글로벌 마스크 (전체 토큰 attending)

        # (3) MHA
        attn_output = self.attention(
            hidden_states,
            mask=attention_mask,
            rope_embeds=rope_embeds,
            training=training,
        )
        hidden_states = hidden_states + attn_output

        # (4) MLP
        normed = self.mlp_norm(hidden_states)
        mlp_out = self.Wi(normed)
        # GLU
        x_in, gate = tf.split(mlp_out, 2, axis=-1)
        # x_in = self.gelu_approx(x_in)
        x_in = tf.nn.gelu(x_in)
        mlp_out = x_in * gate
        if training:
            mlp_out = self.output_dropout(mlp_out, training=training)
        mlp_out = self.Wo(mlp_out)
        hidden_states = hidden_states + mlp_out

        return hidden_states


class ModernGTETensorFlow(tf.keras.Model):
    def __init__(
        self,
        model_name,
        normalize_embeddings=False,
        use_fp16=True,
        query_instruction_for_retrieval=None,
        query_instruction_format="{}{}",
        pooling_method="cls",
        trust_remote_code=False,
        cache_dir=None,
        batch_size=256,
        query_max_length=512,
        passage_max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        dropout_rate=0.1,
    ):
        super().__init__(name="bge-m3-tensorflow")

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.query_instruction_for_retrieval = query_instruction_for_retrieval
        self.query_instruction_format = query_instruction_format
        self.pooling_method = pooling_method
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.return_dense = return_dense
        self.return_sparse = return_sparse
        self.return_colbert_vecs = return_colbert_vecs
        self.dropout_rate = dropout_rate

        # 로딩
        self.config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

        # ModernBertConfig와 유사한 속성들 예시로 옮겨둠
        self.d_model = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.num_layers = self.config.num_hidden_layers
        self.vocab_size = self.config.vocab_size

        # global / local 몇 번째 레이어마다?
        # 예시: self.config.global_attn_every_n_layers = 3 (원 ModernBert 기본 예시)
        if not hasattr(self.config, "global_attn_every_n_layers"):
            self.config.global_attn_every_n_layers = 3

        # local_attention 크기 지정 (원 ModernBert에선 config.local_attention=128 같은 식)
        if not hasattr(self.config, "local_attention"):
            self.config.local_attention = 128
        # rope_theta
        if not hasattr(self.config, "global_rope_theta"):
            self.config.global_rope_theta = 10000.0
        if not hasattr(self.config, "local_rope_theta"):
            self.config.local_rope_theta = None

        self._build_embeddings()
        self._build_encoder_layers()
        self._build_pooler()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, cache_dir=cache_dir
        )

    def _build_embeddings(self):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="embeddings",
                shape=[self.vocab_size, self.d_model],
                initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            )

        self.layerNorm = tf.keras.layers.LayerNormalization(epsilon=1e-5, center=False)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

    def _build_encoder_layers(self):
        self.encoder_layers = []
        for i in range(self.num_layers):
            layer = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.dropout_rate,
                name=f"encoder.layer.{i}",
                layer_id=i,
                config=self.config,
            )
            self.encoder_layers.append(layer)

    def _build_pooler(self):
        self.final_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, center=False)

    def call(self, inputs, training=False, output_hidden_states=False):
        """
        inputs: {
            'input_ids': (B, S),
            'attention_mask': (B, S)
        }
        """
        input_ids = tf.cast(inputs["input_ids"], tf.int32)
        attention_mask_2d = tf.cast(inputs["attention_mask"], TFDTYPE)  # (B,S)

        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]

        # PyTorch처럼 (B,S)->(B,1,S) -> (B,1,S,S)
        #  1. padding인 곳은 0 => -∞
        #  2. 유효 토큰=1 => 0.0
        #  ==> BERT는 causal X, 그냥 "패딩 무시" 목적
        # shaped_mask: [B,1,S]
        shaped_mask = attention_mask_2d[:, None, :]
        # -> broadcast => [B, S, S]
        shaped_mask = tf.tile(shaped_mask, [1, seq_len, 1])  # [B,S,S]
        # => 1이면 0.0, 0이면 -∞
        zeros_ = tf.zeros_like(shaped_mask, dtype=TFDTYPE)
        neg_inf_ = tf.fill(tf.shape(shaped_mask), TFDTYPE.min)  # -1e9)
        shaped_mask = tf.where(tf.equal(shaped_mask, 1.0), zeros_, neg_inf_)
        # 최종 4D
        shaped_mask = shaped_mask[:, None, :, :]  # [B,1,S,S]

        # 임베딩
        inputs_embeds = tf.gather(self.weight, input_ids)  # [B,S,d_model]
        hidden_states = self.layerNorm(inputs_embeds)

        if training:
            hidden_states = self.dropout(hidden_states, training=training)

        all_hidden_states = [hidden_states] if output_hidden_states else []

        # Encoder
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, attention_mask=shaped_mask, training=training
            )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        # final norm
        pooled_output = self.final_norm(hidden_states)

        # cls pooling
        if self.pooling_method == "cls":
            pooled_output = pooled_output[:, 0]  # [B, d_model]

        outputs = {
            "dense_vecs": pooled_output,  # ex) CLS벡터
            "last_hidden_state": hidden_states,
        }
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
        return outputs


def save_model_with_tokenizer(model, tokenizer, save_path):
    """Model + Tokenizer 저장 예시"""
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, "model")

    # 더미 입력으로 build
    dummy_inputs = {
        "input_ids": tf.zeros((2, 12), dtype=tf.int32),
        "attention_mask": tf.ones((2, 12), dtype=tf.int32),
    }
    _ = model(dummy_inputs, training=False, output_hidden_states=True)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec(shape=[None, None], dtype=tf.int32, name="attention_mask"),
        ]
    )
    def serving_fn(input_ids, attention_mask):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = model(inputs=inputs, training=False, output_hidden_states=True)
        # hidden_states까지 반환
        hidden_states = tf.stack(
            outputs["hidden_states"], axis=0
        )  # (num_layers+1, B, S, d_model)
        return {
            "dense_vecs": outputs["dense_vecs"],  # CLS Token
            "hidden_states": hidden_states,
        }

    tf.saved_model.save(
        model, model_save_path, signatures={"serving_default": serving_fn}
    )

    tokenizer.save_pretrained(save_path)
    return model_save_path
