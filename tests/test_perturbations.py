"""Synthetic CPU engineering checks, not corruption robustness research results."""

import copy
import io
import random
import unittest

import numpy as np
from PIL import Image

from pcdc.perturbations import corrupt_rgb, make_injection_mask


CASES = {
    "none": {},
    "brightness": {"factor": 1.3},
    "contrast": {"factor": 0.6, "pivot": 0.5},
    "gamma": {"exponent": 1.4},
    "color_bias": {"offset_rgb": [0.12, -0.08, 0.04]},
    "haze": {"opacity": 0.3, "airlight_rgb": [0.9, 0.95, 1.0]},
    "shadow": {"attenuation": 0.4},
    "noise": {"sigma": 0.06},
    "blur": {"radius": 1.2},
    "jpeg": {"quality": 35, "subsampling": 2},
}

ELLIPSE = {"mode": "ellipse", "center_x": 0.5, "center_y": 0.5,
           "radius_x": 0.36, "radius_y": 0.3, "feather": 0.4}


def quantize(values):
    return np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)


class CorruptionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.Generator(np.random.PCG64(42)).integers(
            0, 256, (17, 19, 3), dtype=np.uint8
        )

    def test_every_kind_is_repeatable_and_does_not_mutate_inputs(self):
        for kind, parameters in CASES.items():
            for mode in ({"mode": "global"}, ELLIPSE):
                with self.subTest(kind=kind, mask=mode["mode"]):
                    image = self.image.copy()
                    mask = make_injection_mask(image.shape[:2], mode)
                    image_before, mask_before = image.copy(), mask.copy()
                    settings_before = copy.deepcopy(parameters)
                    image.flags.writeable = mask.flags.writeable = False
                    first = corrupt_rgb(image, kind, parameters, seed=123, mask=mask)
                    second = corrupt_rgb(image, kind, parameters, seed=123, mask=mask)
                    np.testing.assert_array_equal(first, second)
                    np.testing.assert_array_equal(image, image_before)
                    np.testing.assert_array_equal(mask, mask_before)
                    self.assertEqual(parameters, settings_before)
                    self.assertEqual(first.shape, image.shape)
                    self.assertEqual(first.dtype, np.uint8)
                    self.assertFalse(np.shares_memory(first, image))
                    self.assertTrue(first.flags.writeable)
                    if kind != "none":
                        self.assertTrue(np.any(first != image), kind)

    def test_empty_support_and_neutral_parameters_are_bitwise_identities(self):
        empty = make_injection_mask(self.image.shape[:2], {"mode": "empty"})
        for kind, parameters in CASES.items():
            with self.subTest(kind=kind, path="empty"):
                result = corrupt_rgb(self.image, kind, parameters, seed=8, mask=empty)
                np.testing.assert_array_equal(result, self.image)
                self.assertFalse(np.shares_memory(result, self.image))
        neutral = {
            "none": {}, "brightness": {"factor": 1},
            "contrast": {"factor": 1, "pivot": 0.5}, "gamma": {"exponent": 1},
            "color_bias": {"offset_rgb": [0, 0, 0]},
            "haze": {"opacity": 0, "airlight_rgb": [0.9, 0.9, 0.9]},
            "shadow": {"attenuation": 0}, "noise": {"sigma": 0},
            "blur": {"radius": 0},
        }
        for kind, parameters in neutral.items():
            with self.subTest(kind=kind, path="neutral"):
                result = corrupt_rgb(self.image, kind, parameters, seed=9)
                np.testing.assert_array_equal(result, self.image)

    def test_mask_exterior_is_exact_and_none_equals_global(self):
        mask = make_injection_mask(self.image.shape[:2], ELLIPSE)
        global_mask = make_injection_mask(self.image.shape[:2], {"mode": "global"})
        self.assertTrue(np.any(mask == 0))
        self.assertTrue(np.any((mask > 0) & (mask < 1)))
        for kind, parameters in CASES.items():
            with self.subTest(kind=kind):
                masked = corrupt_rgb(self.image, kind, parameters, seed=11, mask=mask)
                np.testing.assert_array_equal(masked[mask == 0], self.image[mask == 0])
                np.testing.assert_array_equal(
                    corrupt_rgb(self.image, kind, parameters, seed=11),
                    corrupt_rgb(self.image, kind, parameters, seed=11, mask=global_mask),
                )

    def test_simple_operators_match_float_formulas(self):
        source = np.array([[[0, 63, 127], [128, 192, 255]]], dtype=np.uint8)
        x = source.astype(np.float64) / 255
        formulas = {
            "brightness": x * 1.3,
            "contrast": (x - 0.5) * 0.6 + 0.5,
            "gamma": x ** 1.4,
            "color_bias": x + [0.12, -0.08, 0.04],
            "haze": x * 0.7 + 0.3 * np.array([0.9, 0.95, 1.0]),
            "shadow": x * 0.6,
        }
        for kind, expected in formulas.items():
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(
                    corrupt_rgb(source, kind, CASES[kind], seed=0), quantize(expected)
                )

    def test_float_blend_precedes_clipping_and_rounding(self):
        source = np.array([[[150, 3, 4]]], dtype=np.uint8)
        mask = np.array([[0.5]], dtype=np.float32)
        # Red transforms to 300 before blending: (150 + 300) / 2 -> 225.
        # Clipping/rounding the transformed image first would give 202 instead.
        result = corrupt_rgb(source, "brightness", {"factor": 2}, seed=0, mask=mask)
        np.testing.assert_array_equal(result, np.array([[[225, 4, 6]]], dtype=np.uint8))
        # 3 -> 3.9, then alpha=0.5 -> 3.45 -> 3. Intermediate rounding to
        # 4 would instead yield 3.5 -> 4, so this checks quantization order.
        source = np.full((1, 1, 3), 3, dtype=np.uint8)
        np.testing.assert_array_equal(
            corrupt_rgb(source, "brightness", {"factor": 1.3}, seed=0, mask=mask), source
        )

    def test_extreme_valid_parameters_clip_safely(self):
        source = np.array([[[255, 0, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(
            corrupt_rgb(source, "brightness", {"factor": np.finfo(float).max}, seed=0),
            np.array([[[255, 0, 255]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            corrupt_rgb(source, "contrast", {"factor": 0, "pivot": 0.5}, seed=0),
            np.full_like(source, 128),
        )
        np.testing.assert_array_equal(
            corrupt_rgb(source, "shadow", {"attenuation": 1}, seed=0),
            np.zeros_like(source),
        )

    def test_noise_uses_local_pcg64_and_seed_changes_output(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        seed = 2**64 + 31
        result = corrupt_rgb(self.image, "noise", {"sigma": 0.06}, seed=seed)
        generator = np.random.Generator(np.random.PCG64(seed))
        expected = self.image.astype(np.float64) / 255 + generator.normal(
            0, 0.06, self.image.shape
        )
        np.testing.assert_array_equal(result, quantize(expected))
        self.assertTrue(np.any(result != corrupt_rgb(
            self.image, "noise", {"sigma": 0.06}, seed=seed + 1
        )))
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])

    def test_nonrandom_operators_ignore_seed_without_mutating_global_rngs(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        for kind, parameters in CASES.items():
            if kind == "noise":
                continue
            with self.subTest(kind=kind):
                np.testing.assert_array_equal(
                    corrupt_rgb(self.image, kind, parameters, seed=0),
                    corrupt_rgb(self.image, kind, parameters, seed=np.int64(783)),
                )
        self.assertEqual(random.getstate(), python_state)
        after = np.random.get_state()
        self.assertEqual(after[0], numpy_state[0])
        np.testing.assert_array_equal(after[1], numpy_state[1])
        self.assertEqual(after[2:], numpy_state[2:])

    def test_blur_matches_reflect_gaussian_without_intermediate_quantization(self):
        source = np.array([[[3, 30, 230], [47, 119, 4], [193, 87, 139]],
                           [[205, 35, 51], [9, 247, 83], [160, 72, 214]]], dtype=np.uint8)
        sigma = 0.7
        half_width = int(np.ceil(3 * sigma))
        indices = np.arange(-half_width, half_width + 1)
        kernel = np.exp(-0.5 * (indices / sigma) ** 2)
        kernel /= kernel.sum()
        expected = np.pad(source.astype(np.float64) / 255,
                          ((half_width, half_width), (half_width, half_width), (0, 0)),
                          mode="reflect")
        filtered = np.empty(source.shape, dtype=np.float64)
        for row in range(source.shape[0]):
            for col in range(source.shape[1]):
                region = expected[row:row + len(kernel), col:col + len(kernel)]
                filtered[row, col] = np.sum(
                    region * kernel[:, None, None] * kernel[None, :, None], axis=(0, 1)
                )
        mask = np.full(source.shape[:2], 0.43, dtype=np.float32)
        x = source.astype(np.float64) / 255
        expected_result = quantize(x * (1 - mask.astype(np.float64)[..., None])
                                   + filtered * mask.astype(np.float64)[..., None])
        np.testing.assert_array_equal(
            corrupt_rgb(source, "blur", {"radius": sigma}, seed=0, mask=mask), expected_result
        )
        # Reflect padding is also well defined for singleton spatial axes.
        constant = np.full((1, 7, 3), 129, dtype=np.uint8)
        np.testing.assert_array_equal(
            corrupt_rgb(constant, "blur", {"radius": 1.2}, seed=0), constant
        )

    def test_jpeg_is_explicit_codec_roundtrip_including_quality_100(self):
        for quality in (1, 35, 100):
            for subsampling in (0, 1, 2):
                with self.subTest(quality=quality, subsampling=subsampling):
                    settings = {"quality": quality, "subsampling": subsampling}
                    buffer = io.BytesIO()
                    Image.fromarray(self.image).save(
                        buffer, format="JPEG", quality=quality, subsampling=subsampling,
                        optimize=False, progressive=False,
                    )
                    buffer.seek(0)
                    with Image.open(buffer) as decoded:
                        expected = np.array(decoded.convert("RGB"))
                    result = corrupt_rgb(self.image, "jpeg", settings, seed=5)
                    np.testing.assert_array_equal(result, expected)
                    self.assertTrue(np.any(result != self.image))
                    mask = np.full(self.image.shape[:2], 0.5, dtype=np.float64)
                    np.testing.assert_array_equal(
                        corrupt_rgb(self.image, "jpeg", settings, seed=5, mask=mask),
                        quantize((self.image.astype(np.float64) / 255) * 0.5
                                 + (expected.astype(np.float64) / 255) * 0.5),
                    )

    def test_invalid_image_seed_mask_and_kind_rejected(self):
        invalid_images = [None, self.image.tolist(), self.image.astype(np.float32),
                          self.image[..., 0], self.image[..., :2],
                          np.zeros((0, 3, 3), dtype=np.uint8),
                          np.zeros((3, 0, 3), dtype=np.uint8)]
        for value in invalid_images:
            with self.subTest(image_type=type(value)), self.assertRaises(ValueError):
                corrupt_rgb(value, "none", {}, seed=0)
        for seed in (-1, 0.0, True, np.bool_(False), "1", None, float("nan"), float("inf")):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                corrupt_rgb(self.image, "none", {}, seed=seed)
        bad_masks = [0.5, [[1.0]], np.ones(self.image.shape[:2], dtype=np.uint8),
                     np.ones(self.image.shape[:2], dtype=bool),
                     np.ones((*self.image.shape[:2], 1), dtype=np.float32)]
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            bad_masks.append(np.full(self.image.shape[:2], value, dtype=np.float32))
        for mask in bad_masks:
            with self.subTest(mask_type=type(mask)), self.assertRaises(ValueError):
                corrupt_rgb(self.image, "none", {}, seed=0, mask=mask)
        for kind in ("rotate", "BRIGHTNESS", "", None, 1, []):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                corrupt_rgb(self.image, kind, {}, seed=0)

    def test_parameter_schema_and_ranges_rejected_even_with_empty_support(self):
        empty = np.zeros(self.image.shape[:2], dtype=np.float32)
        invalid = [
            ("none", {"factor": 1}), ("none", None), ("none", []),
            ("brightness", {}), ("brightness", {"factor": 0}),
            ("brightness", {"factor": -1}), ("brightness", {"factor": True}),
            ("brightness", {"factor": "1.0"}), ("brightness", {"factor": float("nan")}),
            ("contrast", {"factor": 1}), ("contrast", {"factor": -1, "pivot": 0.5}),
            ("contrast", {"factor": 1, "pivot": 0.4}),
            ("gamma", {"exponent": 0}), ("gamma", {"exponent": float("inf")}),
            ("color_bias", {"offset_rgb": [0, 0]}),
            ("color_bias", {"offset_rgb": [0, 0, 1.1]}),
            ("color_bias", {"offset_rgb": [0, False, 0]}),
            ("haze", {"opacity": 1.01, "airlight_rgb": [1, 1, 1]}),
            ("haze", {"opacity": 0, "airlight_rgb": [1, -0.1, 1]}),
            ("shadow", {"attenuation": -0.01}), ("shadow", {"attenuation": 1.01}),
            ("noise", {"sigma": -0.01}), ("noise", {"sigma": 1.01}),
            ("blur", {"radius": -1}), ("blur", {"radius": float("inf")}),
            ("jpeg", {"quality": 0, "subsampling": 2}),
            ("jpeg", {"quality": 101, "subsampling": 2}),
            ("jpeg", {"quality": 30.0, "subsampling": 2}),
            ("jpeg", {"quality": True, "subsampling": 2}),
            ("jpeg", {"quality": 30, "subsampling": 3}),
            ("jpeg", {"quality": 30, "subsampling": "4:2:0"}),
            ("jpeg", {"quality": 30}),
        ]
        for kind, parameters in CASES.items():
            invalid.append((kind, dict(parameters, unrecognized=0)))
        for kind, parameters in invalid:
            with self.subTest(kind=kind, parameters=parameters), self.assertRaises(ValueError):
                corrupt_rgb(self.image, kind, parameters, seed=0, mask=empty)


class InjectionMaskTests(unittest.TestCase):
    def test_modes_shape_dtype_and_repeatability(self):
        for mode, expected in (("global", 1), ("empty", 0)):
            result = make_injection_mask((5, 9), {"mode": mode})
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, np.full((5, 9), expected, dtype=np.float32))
        before = copy.deepcopy(ELLIPSE)
        result = make_injection_mask((17, 19), ELLIPSE)
        np.testing.assert_array_equal(result, make_injection_mask((17, 19), ELLIPSE))
        self.assertEqual(ELLIPSE, before)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        self.assertTrue(np.any(result == 0))
        self.assertTrue(np.any(result == 1))
        self.assertTrue(np.any((result > 0) & (result < 1)))

    def test_ellipse_uses_pixel_centers_and_explicit_xy_axes(self):
        spec = {"mode": "ellipse", "center_x": 0.5, "center_y": 0.5,
                "radius_x": 0.4, "radius_y": 0.25, "feather": 0.5}
        result = make_injection_mask((3, 5), spec)
        self.assertEqual(result[1, 2], 1)
        np.testing.assert_array_equal(result[0], np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(result[2], np.zeros(5, dtype=np.float32))
        self.assertEqual(result[1, 0], 0)
        self.assertEqual(result[1, 4], 0)
        self.assertEqual(result[1, 1], 1)
        self.assertEqual(result[1, 3], 1)
        tiny = dict(spec, radius_x=0.01, radius_y=0.01)
        np.testing.assert_array_equal(make_injection_mask((2, 2), tiny), np.zeros((2, 2)))

    def test_invalid_shapes_modes_schema_and_ranges_rejected(self):
        for shape in ([4, 5], (4,), (4, 5, 3), (0, 5), (4, -1), (True, 4), (4.0, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                make_injection_mask(shape, {"mode": "global"})
        bad = [None, {}, [], {"mode": 1}, {"mode": "label"},
               {"mode": "global", "semantic_mask": []}, {"mode": "empty", "seed": 1},
               {"mode": "ellipse"}, dict(ELLIPSE, random_key=3)]
        for key, value in (("center_x", -0.1), ("center_y", 1.1), ("radius_x", 0),
                           ("radius_y", -0.1), ("radius_y", 1.1), ("feather", 0),
                           ("feather", 1.1), ("center_x", float("nan")),
                           ("radius_x", float("inf")), ("feather", True)):
            bad.append(dict(ELLIPSE, **{key: value}))
        for spec in bad:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                make_injection_mask((4, 5), spec)


if __name__ == "__main__":
    unittest.main()
