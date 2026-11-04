import numpy as np

from cii.registration import Transform, fit_affine, fit_similarity, leave_one_out, residuals


def _true_transform(mirror=True, scale=1.0012, rot_deg=-0.4, t=(560.0, -2100.0)):
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = np.diag([-1.0, 1.0]) if mirror else np.eye(2)
    return Transform(scale * R @ M, np.array(t))


def _points():
    return np.array([[-2870.0, 2685.0], [3370.0, 2785.0], [-4039.0, 7736.0], [3057.0, 7182.0]])


def test_similarity_recovers_mirrored_transform():
    T0 = _true_transform()
    src = _points()
    dst = np.column_stack(T0.apply(src[:, 0], src[:, 1]))
    T = fit_similarity(src, dst)
    assert T.mirrored
    assert np.isclose(T.scale, 1.0012, atol=1e-6)
    assert np.isclose(T.rotation_deg, -0.4, atol=1e-6)
    assert np.abs(residuals(T, src, dst)).max() < 1e-6


def test_inverse_and_json(tmp_path):
    T = _true_transform()
    x, y = T.apply(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    xi, yi = T.inverse().apply(x, y)
    assert np.allclose(xi, [1, 2]) and np.allclose(yi, [3, 4])
    p = tmp_path / "t.json"
    T.to_json(p)
    T2 = Transform.from_json(p)
    assert np.allclose(T2.A, T.A) and np.allclose(T2.t, T.t)


def test_affine_and_leave_one_out_with_noise():
    rng = np.random.default_rng(0)
    T0 = _true_transform()
    src = _points()
    dst = np.column_stack(T0.apply(src[:, 0], src[:, 1])) + rng.normal(0, 1.0, (4, 2))
    assert np.abs(residuals(fit_affine(src, dst), src, dst)).max() < 3
    assert np.hypot(*leave_one_out(src, dst).T).max() < 10
