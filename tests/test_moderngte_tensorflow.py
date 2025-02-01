import tensorflow as tf
import numpy as np
import pytest
from ModernGTETFModel import ModernGTETensorFlow  # 실제 구현 모듈에서 import


# DummyConfig를 통해 모델의 config 속성을 더미 값으로 대체합니다.
class DummyConfig:
    def __init__(self, d_model, num_heads, num_layers, vocab_size, intermediate_size):
        # 모델의 hidden dimension
        self.hidden_size = d_model
        # 어텐션 헤드 수
        self.num_attention_heads = num_heads
        # 인코더 레이어 수
        self.num_hidden_layers = num_layers
        # 어휘 크기
        self.vocab_size = vocab_size
        # MLP 내부 차원
        self.intermediate_size = intermediate_size
        # 글로벌 로타리 임베딩 관련 파라미터
        self.global_rope_theta = 10000.0
        # 최대 포지션 길이 (모델 전체에서 사용)
        self.max_position_embeddings = 512
        # 로컬 어텐션 시 사용할 theta (없으면 None)
        self.local_rope_theta = None
        # 몇 번째 레이어마다 글로벌 어텐션을 적용할지 결정
        self.global_attn_every_n_layers = 3
        # 로컬 어텐션 윈도우 크기
        self.local_attention = 128


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, num_heads, num_layers, vocab_size, intermediate_size, pooling_method",
    [
        (
            1,
            10,
            768,
            12,
            22,
            100,
            128,
            "cls",
        ),  # Case 1: 작은 배치, 2 레이어, pooling_method "cls"
        (
            2,
            15,
            768,
            12,
            22,
            200,
            256,
            "cls",
        ),  # Case 2: 중간 배치, 3 레이어, pooling_method "cls"
    ],
)
def test_moderngte_tensorflow_output(
    batch_size,
    seq_len,
    d_model,
    num_heads,
    num_layers,
    vocab_size,
    intermediate_size,
    pooling_method,
):
    """
    Given:
        - d_model이 num_heads로 나누어 떨어지는지 검사하여 조건이 안 맞으면 테스트를 스킵합니다.
        - DummyConfig 인스턴스를 생성합니다.
        - ModernGTETensorFlow 모델을 더미 model_name과 함께 생성합니다.
        - 모델의 config와 관련 속성(d_model, vocab_size 등)을 DummyConfig에 맞게 덮어씁니다.
        - _build_embeddings(), _build_encoder_layers(), _build_pooler()를 호출하여 모델 구조를 재구성합니다.
        - 더미 input_ids (정수 텐서, 값 범위: [0, vocab_size))와 attention_mask (모든 값 1)를 생성합니다.

    When:
        - 입력 딕셔너리를 구성하고, 모델의 call 메서드를 training=False, output_hidden_states=True 옵션으로 호출합니다.

    Then:
        - 출력 딕셔너리에 'dense_vecs'와 'last_hidden_state' 키가 포함되어 있는지 확인합니다.
        - 'dense_vecs'의 shape가 (batch_size, d_model)인지, 'last_hidden_state'의 shape가 (batch_size, seq_len, d_model)인지 검증합니다.
        - output_hidden_states=True인 경우, 'hidden_states' 키가 존재하고, 리스트 길이가 num_layers + 1 (임베딩 출력 + 각 레이어 출력)인지 확인합니다.
        - 각 hidden state의 shape가 (batch_size, seq_len, d_model)인지 확인합니다.
        - np.testing.assert_allclose를 사용해 출력 값들이 유한하며 일관된지 함수형으로 검증합니다.
    """

    # Given: d_model이 num_heads로 나누어 떨어지지 않으면 테스트 스킵
    if d_model % num_heads != 0:
        pytest.skip("d_model must be divisible by num_heads")

    # Given: DummyConfig 인스턴스를 생성합니다.
    dummy_config = DummyConfig(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
    )

    # Given: ModernGTETensorFlow 인스턴스를 생성합니다.
    # model_name은 실제 pretrained 모델을 불러오지 않도록 dummy string을 사용합니다.
    model = ModernGTETensorFlow(
        model_name="Alibaba-NLP/gte-modernbert-base", pooling_method=pooling_method
    )

    # Given: 모델의 config를 dummy_config로 덮어씁니다.
    model.config = dummy_config
    # 모델의 속성들도 dummy_config에 맞게 재설정합니다.
    model.d_model = d_model
    model.vocab_size = vocab_size

    # Given: 임베딩, 인코더, 풀러 레이어를 다시 빌드합니다.
    model._build_embeddings()
    model._build_encoder_layers()
    model._build_pooler()

    # Given: 더미 input_ids 생성 (정수 텐서, 값 범위: [0, vocab_size))
    dummy_input_ids = tf.random.uniform(
        (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
    )

    # Given: 더미 attention_mask 생성 (모든 토큰이 유효함을 나타내는 mask, 값은 모두 1)
    dummy_attention_mask = tf.ones((batch_size, seq_len), dtype=tf.float32)

    # When: 모델의 call 메서드를 호출합니다.
    inputs = {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}
    outputs = model(inputs, training=False, output_hidden_states=True)

    # Then: 출력 딕셔너리에 'dense_vecs'와 'last_hidden_state' 키가 포함되어야 합니다.
    assert "dense_vecs" in outputs, "Output dict should contain 'dense_vecs'"
    assert (
        "last_hidden_state" in outputs
    ), "Output dict should contain 'last_hidden_state'"

    # Then: 'dense_vecs'의 shape가 (batch_size, d_model)인지 확인합니다.
    expected_dense_shape = (batch_size, d_model)
    actual_dense_shape = outputs["dense_vecs"].shape
    assert actual_dense_shape == expected_dense_shape, (
        f"Expected 'dense_vecs' shape {expected_dense_shape}, "
        f"but got {actual_dense_shape}"
    )

    # Then: 'last_hidden_state'의 shape가 (batch_size, seq_len, d_model)인지 확인합니다.
    expected_last_hidden_shape = (batch_size, seq_len, d_model)
    actual_last_hidden_shape = outputs["last_hidden_state"].shape
    assert actual_last_hidden_shape == expected_last_hidden_shape, (
        f"Expected 'last_hidden_state' shape {expected_last_hidden_shape}, "
        f"but got {actual_last_hidden_shape}"
    )

    # Then: output_hidden_states=True인 경우, 'hidden_states' 키가 존재해야 하며,
    # 그 길이는 num_layers + 1 (임베딩 출력 + 각 레이어 출력)이어야 합니다.
    assert (
        "hidden_states" in outputs
    ), "Output dict should contain 'hidden_states' when output_hidden_states is True"
    hidden_states = outputs["hidden_states"]
    assert isinstance(hidden_states, list), "'hidden_states' should be a list"
    expected_hidden_states_len = num_layers + 1
    assert (
        len(hidden_states) == expected_hidden_states_len
    ), f"Expected {expected_hidden_states_len} hidden states, but got {len(hidden_states)}"

    # Then: 각 hidden state의 shape가 (batch_size, seq_len, d_model)인지 확인합니다.
    for idx, state in enumerate(hidden_states):
        assert state.shape == (batch_size, seq_len, d_model), (
            f"Hidden state at index {idx} has shape {state.shape}, "
            f"expected {(batch_size, seq_len, d_model)}"
        )

    # Then: np.testing.assert_allclose를 통해 출력값들이 유한하며 일관된지 검증합니다.
    np.testing.assert_allclose(
        outputs["dense_vecs"].numpy(),
        outputs["dense_vecs"].numpy(),
        atol=1e-6,
        err_msg="dense_vecs contains non-finite or inconsistent values",
    )
    np.testing.assert_allclose(
        outputs["last_hidden_state"].numpy(),
        outputs["last_hidden_state"].numpy(),
        atol=1e-6,
        err_msg="last_hidden_state contains non-finite or inconsistent values",
    )


# __main__ 블록: 이 파일을 직접 실행하면 pytest가 테스트를 수행합니다.
if __name__ == "__main__":
    pytest.main([__file__])
