import tensorflow as tf
import numpy as np
import pytest

from ModernGTETFModel import create_local_sliding_window_mask


@pytest.mark.parametrize(
    "batch_size, seq_len, window_size",
    [
        (1, 10, 4),  # batch_size 1, sequence_length 10, window_size 4
        (2, 15, 6),  # batch_size 2, sequence_length 15, window_size 6
        (3, 20, 8),  # batch_size 3, sequence_length 20, window_size 8
    ],
)
def test_create_local_sliding_window_mask(
    batch_size: int, seq_len: int, window_size: int
):
    ATTENTION_MASK_DIMENSION: int = 1
    HALF_WINDOW_SIZE: int = window_size // 2

    # Given: 글로벌 마스크가 주어졌습니다. 모든 값이 0.0인 텐서입니다. shape : (batch_size, 1, seq_len, seq_len)
    # Q: 왜 seq_len 은 2번 반복할까? --- A: 답변해보기 (어텐션 계산 시에 각 토큰이 다른 토큰들과의 패딩인지, 유효한지 관계를 보기 위함이다.
    # 다시 말해서, seq_len, seq_len -> 요거는 2D 마스크이다.
    # tf.zeros 는 이유는 ? 어텐션 마스크에서는 valid 위치는 보통 0.0으로, 마스킹해야 하는 경우에는 -∞ 값으로 설정하게 된다.
    # 여기에서는, 모든 위치가 유효하다고 전제 하므로 모든 원소를 0.0 으로 초기화 한다.
    global_mask: tf.zeros = tf.zeros(
        (batch_size, ATTENTION_MASK_DIMENSION, seq_len, seq_len), dtype=tf.float32
    )

    # When: 함수를 호출하여 슬라이딩 윈도우 마스크를 생성합니다.
    result: tf.Tensor = create_local_sliding_window_mask(global_mask, window_size)

    # Then: tensor 의 shape 가 (batch_size, 1, seq_len, seq_len) 인지를 검증해봅니다.
    assert result.shape == (batch_size, ATTENTION_MASK_DIMENSION, seq_len, seq_len)

    # Then: 각 (i, j) 위치에 대하여 윈도우의 내부이면 0.0 이요, 그 외부이면 매우 작은 값이 나온다는 것을 검증 합니다.
    result_np: np.ndarray = result.numpy()

    # 인덱스 배열을 생성한 후에, 각 위치에 저장된 값의 절댓값 을 계산 합니다.
    # 절댓값 행렬 (seq_len, seq_len) 은 두 1D 배열 간의 외적 차이를 계산하여 각 토큰 위치 간의 거리를 나타냅니다.
    indices: np.ndarray = np.arange(seq_len)
    diff_matrix: np.ndarray = np.abs(np.subtract.outer(indices, indices))

    # 절대값의 차이가 half_w 이하인 윈도우 내부 위치는 True이고, 외부는 False인 마스크를 만들어봅니다.
    valid_mask: np.ndarray = diff_matrix <= HALF_WINDOW_SIZE

    # 결과 행렬의 첫 번째 배치의 값을 추출해봅니다. shape 는 (seq_len, seq_len) 이 됩니다.
    # 추출한 이후에는 윈도우 내부의 값과 윈도우 외부의 값을 가져와봅니다.
    result_matrix: np.ndarray = result_np[0, 0]
    window_inner_matrix: np.ndarray = result_matrix[valid_mask]
    window_outer_matrix: np.ndarray = result_matrix[~valid_mask]

    # 내부 값들은 0.0 이어야 합니다.
    np.testing.assert_allclose(
        actual=window_inner_matrix,
        desired=0.0,
        atol=1e-6,
        err_msg="윈도우 내부 matrix에서 0.0이 아닌 값이 존재하네요.",
    )

    # 외부 값들은 매우 작은 값 이어야 합니다. 여기서 매우 작은 값은 -1e8 미만을 의미 합니다.
    LOW_THRESHOLD: float = -1e8
    assert np.all(
        outside_values < LOW_THRESHOLD for outside_values in window_outer_matrix
    )


if __name__ == "__main__":
    pytest.main([__file__])
