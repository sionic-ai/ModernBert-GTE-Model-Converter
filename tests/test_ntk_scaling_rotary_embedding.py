"""
로타리 포지셔널 임베딩(Rotary Positional Embedding):
기존 Transformer 모델에서는 토큰의 순서 정보를 절대 위치 임베딩(absolute positional embedding)으로 처리하지만...
로타리 임베딩은 쿼리와 키 벡터에 회전(rotation) 연산을 적용하여 상대적인 위치 정보를 자연스럽게 반영합니다.
이 방식은 특히 긴 문맥을 처리할 때나 상대적 위치 정보를 보다 세밀하게 반영하고 싶을 때 유용합니다.

NTK(Natural Tangent Kernel) 스케일링은 모델의 학습 안정성을 높이고, 네트워크의 스케일에 맞춰 임베딩 값을 조정하는 기법입니다.
즉, NTKScalingRotaryEmbedding은 기본 로타리 임베딩 방식에 NTK 스케일링을 더해, 임베딩이 더 안정적이고 효과적으로 학습되도록 도와줍니다.

이 클래스는 주어진 임베딩 차원(dim), 최대 위치(max_position_embeddings), base 값, 그리고 scaling factor 등의 파라미터를 바탕으로 각 위치에 대한 cosine과 sine 값을 미리 계산하여 캐싱(cache)합니다.
그런 다음 MultiHeadAttention 같은 모듈에서 쿼리와 키 벡터에 이 cos, sin 값을 적용(rotate)함으로써, 토큰 간의 상대적 위치 정보를 어텐션 계산에 반영하게 됩니다.
"""

# tests/test_ntk_scaling_rotary_embedding_bdd.py

import tensorflow as tf  # TensorFlow 라이브러리 임포트 (텐서 연산용)
import numpy as np  # NumPy 임포트 (수치 비교용)
import pytest  # pytest 프레임워크 임포트
from ModernGTETFModel import NTKScalingRotaryEmbedding


@pytest.mark.parametrize(
    "dim, max_pos, base, scaling_factor, test_seq_len",
    [
        (
            32,
            50,
            160000.0,
            1.0,
            30,
        ),  # 케이스 1: test_seq_len(30)이 max_pos(50)보다 작음 → 캐시된 값 사용 기대
        (
            32,
            50,
            160000.0,
            1.0,
            60,
        ),  # 케이스 2: test_seq_len(60)이 max_pos(50)보다 큼 → 새로운 캐시 계산됨
        (
            64,
            100,
            160000.0,
            1.0,
            80,
        ),  # 케이스 3: test_seq_len(80)이 max_pos(100) 이하 → 캐시된 값 사용 기대
    ],
)
def test_ntk_scaling_rotary_embedding_shapes(
    dim, max_pos, base, scaling_factor, test_seq_len
):
    """
    Given a NTKScalingRotaryEmbedding layer instantiated with specific parameters,
      and given a dummy input tensor with a specified sequence length,
    When the layer is called with that input and seq_len is provided,
    Then the returned cos and sin tensors should have shape (seq_len, dim),
      and if seq_len <= max_position_embeddings, they should match the cached values.
    """
    # Given: NTKScalingRotaryEmbedding 인스턴스를 생성합니다.
    rotary_layer = NTKScalingRotaryEmbedding(
        dim=dim,  # 임베딩 차원
        max_position_embeddings=max_pos,  # 최대 position embeddings
        base=base,  # base 값 (예: 160000.0)
        scaling_factor=scaling_factor,  # 스케일링 팩터 (예: 1.0)
    )

    # Given: 더미 입력 텐서를 생성합니다. 실제 값은 중요하지 않고, 단지 seq_len 정보를 전달하기 위한 용도입니다.
    dummy_input = tf.random.uniform((1, test_seq_len, dim), dtype=tf.float32)

    # When: NTKScalingRotaryEmbedding의 call 메서드를 호출하여, cos와 sin 값을 계산합니다.
    cos, sin = rotary_layer(dummy_input, seq_len=test_seq_len)

    # Then: 반환된 cos 텐서의 shape가 (test_seq_len, dim)인지 검증합니다.
    assert cos.shape == (
        test_seq_len,
        dim,
    ), f"Expected cos shape ({test_seq_len}, {dim}), got {cos.shape}"
    # Then: 반환된 sin 텐서의 shape가 (test_seq_len, dim)인지 검증합니다.
    assert sin.shape == (
        test_seq_len,
        dim,
    ), f"Expected sin shape ({test_seq_len}, {dim}), got {sin.shape}"

    # Then: test_seq_len이 max_pos 이하인 경우, 캐시된 cos와 sin의 슬라이스와 동일해야 합니다.
    if test_seq_len <= max_pos:
        # 캐시된 cos 값을 test_seq_len 만큼 슬라이스합니다.
        cached_cos = rotary_layer.cos_cached[:test_seq_len]

        # 캐시된 sin 값을 test_seq_len 만큼 슬라이스합니다.
        cached_sin = rotary_layer.sin_cached[:test_seq_len]

        # np.testing.assert_allclose를 사용하여 계산된 cos와 캐시된 cos가 거의 동일한지 확인합니다.
        np.testing.assert_allclose(
            cos.numpy(),
            cached_cos.numpy(),
            atol=1e-6,
            err_msg="cos values do not match cached values when seq_len <= max_pos",
        )

        # np.testing.assert_allclose를 사용하여 계산된 sin와 캐시된 sin가 거의 동일한지 확인합니다.
        np.testing.assert_allclose(
            sin.numpy(),
            cached_sin.numpy(),
            atol=1e-6,
            err_msg="sin values do not match cached values when seq_len <= max_pos",
        )
    else:
        # test_seq_len이 max_pos보다 클 경우, 새로운 캐시가 생성되므로 캐시와 비교할 수 없습니다.
        # 대신, 계산된 cos와 sin 값이 모두 유한한(finite) 값인지 확인합니다.
        assert np.all(
            np.isfinite(cos.numpy())
        ), "cos contains non-finite values for seq_len > max_pos"
        assert np.all(
            np.isfinite(sin.numpy())
        ), "sin contains non-finite values for seq_len > max_pos"


if __name__ == "__main__":
    pytest.main([__file__])
