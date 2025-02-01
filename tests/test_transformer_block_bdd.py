import tensorflow as tf
import numpy as np
import pytest
from ModernGTETFModel import TransformerBlock


class DummyConfig:
    def __init__(self, d_model, num_heads, intermediate_size):
        self.hidden_size = d_model
        self.num_attention_heads = num_heads
        self.intermediate_size = intermediate_size
        self.global_rope_theta = 10000.0  # 글로벌 로타리 임베딩 base 값
        self.max_position_embeddings = 512  # 전체 최대 시퀀스 길이
        self.local_rope_theta = None  # 로컬 어텐션 시 별도 값 없으면 None
        self.global_attn_every_n_layers = 2  # 예: 2번째 레이어마다 글로벌 어텐션
        self.local_attention = 128  # 로컬 어텐션 윈도우 크기


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads, intermediate_size, dropout_rate, layer_id",
    [
        (
            1,
            10,
            64,
            8,
            128,
            0.0,
            0,
        ),  # Case 1: 작은 배치, dropout 없이, layer_id=0 (글로벌 어텐션 적용)
        (
            2,
            15,
            128,
            8,
            256,
            0.0,
            1,
        ),  # Case 2: 중간 배치, dropout 없이, layer_id=1 (로컬 어텐션 적용)
        (
            3,
            20,
            64,
            4,
            128,
            0.1,
            1,
        ),  # Case 3: dropout 적용, training=False로 실행하여도 dropout layer는 build됨
    ],
)
def test_transformer_block_output_shape(
    batch_size, seq_len, d_model, num_heads, intermediate_size, dropout_rate, layer_id
):
    """
    Given a dummy input tensor of shape (batch_size, seq_len, d_model)
      and a DummyConfig with the specified parameters,
    When the TransformerBlock is invoked with a simple attention mask (all ones),
    Then the output tensor should have the same shape as the input tensor.
    """
    # Given: d_model이 num_heads로 나누어 떨어지는지 확인합니다.
    if d_model % num_heads != 0:
        pytest.skip("d_model must be divisible by num_heads")

    # Given: DummyConfig 인스턴스를 생성합니다.
    config = DummyConfig(d_model, num_heads, intermediate_size)

    # Given: TransformerBlock 인스턴스를 생성합니다.
    # layer_id에 따라 내부에서 글로벌/로컬 설정이 달라집니다.
    transformer_block: TransformerBlock = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        dropout_rate=dropout_rate,
        layer_id=layer_id,
        config=config,
    )

    # Given: 모델의 테스트용 더미 입력(dummy input)을 생성합니다.
    # tf.random.uniform 함수를 사용하여, 지정한 shape인 (batch_size, seq_len, d_model)에 맞게 균등 분포의 무작위 값들을 가진 텐서를 만듭니다.
    # 여기서 batch_size는 한 번에 처리할 데이터 개수, seq_len은 문장(또는 토큰 시퀀스)의 길이, d_model은 각 토큰의 임베딩 차원을 의미합니다.
    dummy_input = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)

    # Given: 단순 attention mask를 생성합니다.
    # 이 부분에서는 모든 토큰이 유효하다는 가정 하에 attention mask를 만듭니다.
    # tf.ones를 사용하여, shape이 (batch_size, seq_len)인 2차원 텐서를 생성합니다.
    # 각 요소의 값은 1로 채워지는데, 이는 해당 위치의 토큰이 “유효(valid)”하다는 의미입니다.
    # 즉, 이 mask는 패딩 토큰이 없어서 모든 토큰에 대해 어텐션을 수행한다는 전제입니다.
    attn_mask_2d = tf.ones((batch_size, seq_len), dtype=tf.float32)

    # 이를 [batch_size, 1, seq_len]로 확장한 뒤, [batch_size, 1, seq_len, seq_len]로 타일링합니다.
    # 타일링할 때는 새로 추가한 차원을 기준으로, 텐서를 반복(tile)합니다.
    # [1, seq_len, 1] 인자는 첫 번째 차원은 그대로, 두 번째 차원을 seq_len 만큼 복제하여 (batch_size, seq_len, seq_len)이 되도록 합니다.
    shaped_mask = tf.tile(tf.expand_dims(attn_mask_2d, axis=1), [1, seq_len, 1])

    # 다시 한 번 차원을 확장하여 최종적으로 shape을 (batch_size, 1, seq_len, seq_len)로 만듭니다.
    shaped_mask = tf.expand_dims(shaped_mask, axis=1)

    # mask 값: 유효 위치는 0.0, 패딩은 -∞; 여기선 모두 유효하므로 0.0
    # shaped_mask와 동일한 shape의 텐서를 생성하되, 모든 값을 0.0으로 채웁니다. 이는 유효한 토큰 위치에 사용됩니다.
    zeros_mask = tf.zeros_like(shaped_mask, dtype=tf.float32)

    # tf.fill을 이용하여, shaped_mask와 동일한 shape의 텐서를 생성하고 모든 값을 tf.float32.min으로 채웁니다.
    # tf.float32.min은 float32 타입에서 표현 가능한 매우 작은 값(실질적으로 -∞에 가까운 값)을 의미합니다.
    # 이는 패딩 토큰 위치에 사용되어, softmax 계산 시 해당 위치의 attention score를 0으로 만들도록 합니다.
    neg_inf_mask = tf.fill(tf.shape(shaped_mask), tf.float32.min)

    # tf.where 함수를 이용하여, shaped_mask의 각 위치가 1.0이면 zeros_mask의 해당 위치 값을 선택하고, 그렇지 않으면 neg_inf_mask의 값을 선택합니다.
    # 여기서는 shaped_mask가 모두 1로 채워져 있기 때문에, 최종적으로 모든 위치가 0.0이 됩니다.
    # 결과적으로, 이 simple_mask는 모든 토큰이 유효하여 패딩이 없음을 나타내며, 어텐션 연산 시 모든 위치에 대해 아무런 마스킹 효과가 없음을 의미합니다.
    simple_mask = tf.where(tf.equal(shaped_mask, 1.0), zeros_mask, neg_inf_mask)

    # When: TransformerBlock의 call 메서드를 호출합니다.
    # training=False로 호출하면 드롭아웃 등이 적용되지 않습니다.
    output = transformer_block(dummy_input, attention_mask=simple_mask, training=False)

    # Then: 출력 텐서의 shape가 입력 텐서와 동일해야 합니다.
    expected_shape = (batch_size, seq_len, d_model)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Then: 함수형 검증 도구를 사용해 출력값이 모두 유한한지 확인합니다.
    # (Residual connection과 MLP 계산 후에도 수치 불안정성이 없어야 함)
    np.testing.assert_allclose(
        output.numpy(),
        output.numpy(),
        atol=1e-6,
        err_msg="Output values are not consistent or contain non-finite numbers.",
    )
