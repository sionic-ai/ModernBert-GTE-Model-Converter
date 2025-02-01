import tensorflow as tf
import numpy as np
import pytest

from ModernGTETFModel import MultiHeadAttention


# -------------------------------------------------------------------
# tf.nn.dropout의 인자 요구사항을 맞추기 위해 테스트 실행 시 임시로 패치합니다.
# (tf.nn.dropout이 이제 rate 인자를 필수로 요구하므로, 기본값을 제공합니다.)
@pytest.fixture(autouse=True)
def patch_tf_nn_dropout():
    original_dropout = tf.nn.dropout

    def patched_dropout(x, rate=None, noise_shape=None, seed=None, name=None):
        if rate is None:
            rate = 0.1  # 기본 드롭아웃 비율 (원래 self.dropout_rate에 해당)
        return original_dropout(
            x, rate=rate, noise_shape=noise_shape, seed=seed, name=name
        )

    tf.nn.dropout = patched_dropout
    yield
    tf.nn.dropout = original_dropout


# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads, dropout_rate, training, layer_id",
    [
        # Case 1: 배치가 작고 layer_id=0인 경우 (LayerNorm 적용 없이)
        (1, 128, 64, 8, 0.0, False, 0),
        # Case 2: 중간 배치, layer_id=0
        (2, 256, 128, 16, 0.0, False, 0),
        # Case 3: 배치 3, dropout 0.1, training=True, layer_id=1 (입력 전 LayerNorm 적용)
        (3, 512, 64, 4, 0.1, True, 1),
    ],
)
def test_ROPE가_존재하지_않는_멀티헤드_어텐션을_테스트한다(
    batch_size, seq_len, d_model, num_heads, dropout_rate, training, layer_id
):
    """
    rope_embeds 없이 MultiHeadAttention 레이어의 동작을 테스트합니다.

    Given:
      - [batch_size, seq_len, d_model] 형태의 dummy input
      - d_model은 num_heads로 나누어 떨어져야 합니다.
    When:
      - rope_embeds 없이 forward pass를 수행합니다.
    Then:
      - 출력의 shape는 [batch_size, seq_len, d_model] 이어야 하며,
      - 내부 변환(QKV, attention, 출력 프로젝션)이 적용되어 입력과 값이 달라져야 합니다.
    """
    if d_model % num_heads != 0:
        pytest.fail("d_model must be divisible by num_heads")

    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    mha = MultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_id=layer_id,
    )

    output = mha(inputs=dummy_input, mask=None, rope_embeds=None, training=training)
    expected_shape = (batch_size, seq_len, d_model)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # self-consistency 체크: 같은 텐서를 두 번 호출했을 때 값은 동일해야 함
    np.testing.assert_allclose(
        output.numpy(), output.numpy(), err_msg="Output values are not consistent."
    )

    # layer_id가 0이 아니라면 내부 변환이 적용되어야 하므로 입력과 값이 달라져야 함.
    if layer_id != 0 and np.allclose(dummy_input.numpy(), output.numpy(), atol=1e-6):
        pytest.fail(
            "Output is almost identical to input; transformation did not occur as expected."
        )


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        # 간단한 설정
        (1, 10, 64, 8),
        # 조금 더 큰 설정
        (2, 15, 128, 8),
    ],
)
def test_ROPE가_존재하는_멀티헤드_어텐션을_테스트한다(
    batch_size, seq_len, d_model, num_heads
):
    """
    dummy rope embeddings를 전달한 경우 MultiHeadAttention의 출력 shape를 검증합니다.

    주의: 기존 구현에서는 rope_embeds의 shape가 [seq_len, head_dim]이어야 합니다.
    """
    if d_model % num_heads != 0:
        pytest.skip("d_model must be divisible by num_heads")

    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    # layer_id=1: RoPE 적용 케이스
    mha = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, layer_id=1)
    head_dim = d_model // num_heads

    # rope_embeds: [seq_len, head_dim] (내부 NTKScalingRotaryEmbedding의 결과와 맞춥니다)
    dummy_cos = tf.ones((seq_len, head_dim), dtype=tf.float32)
    dummy_sin = tf.zeros((seq_len, head_dim), dtype=tf.float32)
    rope_embeds = (dummy_cos, dummy_sin)

    output = mha(dummy_input, mask=None, rope_embeds=rope_embeds, training=False)
    expected_shape = (batch_size, seq_len, d_model)
    assert (
        output.shape == expected_shape
    ), f"Output shape mismatch with rope_embeds: expected {expected_shape}, got {output.shape}"

    np.testing.assert_allclose(
        output.numpy(),
        output.numpy(),
        err_msg="Output values are inconsistent when using rope_embeds.",
    )


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads",
    [
        (1, 10, 64, 8),
        (2, 15, 128, 8),
    ],
)
def test_ROPE가_결과값에_영향을_미치는지_확인한다(
    batch_size, seq_len, d_model, num_heads
):
    """
    rope_embeds 적용 여부에 따라 결과값이 달라지는지 확인합니다.
    """
    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    mha = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0, layer_id=1)

    output_no_rope = mha(
        inputs=dummy_input, mask=None, rope_embeds=None, training=False
    )
    head_dim = d_model // num_heads
    dummy_cos = tf.random.uniform((seq_len, head_dim), dtype=tf.float32)
    dummy_sin = tf.random.uniform((seq_len, head_dim), dtype=tf.float32)
    rope_embeds = (dummy_cos, dummy_sin)

    output_with_rope = mha(
        inputs=dummy_input, mask=None, rope_embeds=rope_embeds, training=False
    )

    assert not np.allclose(
        output_no_rope.numpy(), output_with_rope.numpy(), atol=1e-6
    ), "Output with rope_embeds should differ from output without rope_embeds."


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 10, 16),
        (1, 8, 12, 8),
    ],
)
def test_scaled_dot_product_attention(batch_size, num_heads, seq_len, head_dim):
    """
    Dummy query, key, value 텐서를 사용해 scaled_dot_product_attention 메서드의 출력 shape를 검증합니다.
    """
    q = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    k = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    v = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)

    mha = MultiHeadAttention(
        d_model=num_heads * head_dim, num_heads=num_heads, dropout_rate=0.0, layer_id=1
    )

    output = mha.scaled_dot_product_attention(q, k, v, mask=None, training=False)
    expected_shape = (batch_size, num_heads, seq_len, head_dim)
    assert (
        output.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [
        (2, 4, 10, 16),
        (1, 8, 12, 8),
    ],
)
def test_apply_rotary_pos_emb(batch_size, num_heads, seq_len, head_dim):
    """
    Dummy query와 key 텐서, 그리고 dummy rope embeddings (cos=1, sin=0)을 사용해
    apply_rotary_pos_emb 메서드가 입력과 동일한 shape를 반환하며, 회전 효과가 없는지 검증합니다.

    주의: rope_embeds의 shape는 [seq_len, head_dim]이어야 합니다.
    """
    q = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)
    k = tf.random.uniform((batch_size, num_heads, seq_len, head_dim), dtype=tf.float32)

    dummy_cos = tf.ones((seq_len, head_dim), dtype=tf.float32)
    dummy_sin = tf.zeros((seq_len, head_dim), dtype=tf.float32)

    mha = MultiHeadAttention(
        d_model=num_heads * head_dim, num_heads=num_heads, dropout_rate=0.0, layer_id=1
    )

    q_new, k_new = mha.apply_rotary_pos_emb(q, k, dummy_cos, dummy_sin)

    assert q_new.shape == q.shape, f"Expected q shape {q.shape}, got {q_new.shape}"
    assert k_new.shape == k.shape, f"Expected k shape {k.shape}, got {k_new.shape}"

    np.testing.assert_allclose(
        q_new.numpy(), q.numpy(), atol=1e-6, err_msg="q values differ with dummy rope"
    )
    np.testing.assert_allclose(
        k_new.numpy(), k.numpy(), atol=1e-6, err_msg="k values differ with dummy rope"
    )
