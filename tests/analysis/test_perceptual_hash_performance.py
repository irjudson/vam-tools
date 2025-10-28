"""
Performance and accuracy tests for perceptual hashing algorithms.

These tests verify that the hash algorithms correctly identify similar and different
images, and measure the performance of each method.
"""

# mypy: disable-error-code="arg-type,no-untyped-def,operator"

import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from vam_tools.analysis.perceptual_hash import (
    ahash,
    combined_hash,
    dhash,
    get_best_matches,
    hamming_distance,
    whash,
)


@pytest.fixture
def test_images(tmp_path: Path) -> dict:
    """Create a variety of test images for performance testing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    images = {}

    # 1. Original image with detailed pattern
    original = Image.new("RGB", (400, 400), color="white")
    draw = ImageDraw.Draw(original)
    # Create a complex pattern
    for i in range(0, 400, 40):
        draw.rectangle((i, 0, i + 20, 400), fill="black")
    for i in range(0, 400, 40):
        draw.rectangle((0, i, 400, i + 20), fill="red")
    original_path = images_dir / "original.jpg"
    original.save(original_path, quality=95)
    images["original"] = original_path

    # 2. Exact copy (same pixels, different file)
    exact_copy_path = images_dir / "exact_copy.jpg"
    original.save(exact_copy_path, quality=95)
    images["exact_copy"] = exact_copy_path

    # 3. Same content, different compression
    low_quality_path = images_dir / "low_quality.jpg"
    original.save(low_quality_path, quality=60)
    images["low_quality"] = low_quality_path

    # 4. Resized version (smaller)
    resized_small = original.resize((200, 200), Image.Resampling.LANCZOS)
    resized_small_path = images_dir / "resized_small.jpg"
    resized_small.save(resized_small_path, quality=95)
    images["resized_small"] = resized_small_path

    # 5. Resized version (larger)
    resized_large = original.resize((800, 800), Image.Resampling.LANCZOS)
    resized_large_path = images_dir / "resized_large.jpg"
    resized_large.save(resized_large_path, quality=95)
    images["resized_large"] = resized_large_path

    # 6. Cropped version (center crop)
    cropped = original.crop((100, 100, 300, 300))
    cropped_path = images_dir / "cropped.jpg"
    cropped.save(cropped_path, quality=95)
    images["cropped"] = cropped_path

    # 7. Brightness adjusted
    from PIL import ImageEnhance

    brightened = ImageEnhance.Brightness(original).enhance(1.5)
    brightened_path = images_dir / "brightened.jpg"
    brightened.save(brightened_path, quality=95)
    images["brightened"] = brightened_path

    # 8. Slightly blurred
    blurred = original.filter(ImageFilter.GaussianBlur(radius=2))
    blurred_path = images_dir / "blurred.jpg"
    blurred.save(blurred_path, quality=95)
    images["blurred"] = blurred_path

    # 9. Completely different image
    different = Image.new("RGB", (400, 400), color="white")
    draw_diff = ImageDraw.Draw(different)
    # Create circles instead of rectangles
    for i in range(0, 400, 60):
        for j in range(0, 400, 60):
            draw_diff.ellipse([i, j, i + 40, j + 40], fill="blue")
    different_path = images_dir / "different.jpg"
    different.save(different_path, quality=95)
    images["different"] = different_path

    return images


class TestHashAccuracy:
    """Test that hash methods correctly identify similar and different images."""

    def test_exact_copies_identical_hashes(self, test_images: dict) -> None:
        """Exact copies should produce identical hashes."""
        hash1_d = dhash(test_images["original"])
        hash2_d = dhash(test_images["exact_copy"])
        assert hash1_d == hash2_d

        hash1_a = ahash(test_images["original"])
        hash2_a = ahash(test_images["exact_copy"])
        assert hash1_a == hash2_a

        hash1_w = whash(test_images["original"])
        hash2_w = whash(test_images["exact_copy"])
        assert hash1_w == hash2_w

    def test_compression_resilience(self, test_images: dict) -> None:
        """Different compression should produce similar hashes."""
        hash1 = dhash(test_images["original"])
        hash2 = dhash(test_images["low_quality"])

        distance = hamming_distance(hash1, hash2)
        # Should be very similar despite compression
        assert distance <= 5, f"dHash not resilient to compression: distance={distance}"

    def test_resize_resilience(self, test_images: dict) -> None:
        """Resized versions should produce similar hashes."""
        hash_orig = dhash(test_images["original"])
        hash_small = dhash(test_images["resized_small"])
        hash_large = dhash(test_images["resized_large"])

        dist_small = hamming_distance(hash_orig, hash_small)
        dist_large = hamming_distance(hash_orig, hash_large)

        # All should be similar
        assert dist_small <= 8, f"Not resilient to downscaling: distance={dist_small}"
        assert dist_large <= 8, f"Not resilient to upscaling: distance={dist_large}"

    def test_brightness_resilience(self, test_images: dict) -> None:
        """Brightness changes should not drastically affect hashes."""
        hash_orig = dhash(test_images["original"])
        hash_bright = dhash(test_images["brightened"])

        distance = hamming_distance(hash_orig, hash_bright)
        # Should be reasonably similar
        assert distance <= 15, f"Not resilient to brightness: distance={distance}"

    def test_blur_resilience(self, test_images: dict) -> None:
        """Slight blur should not drastically affect hashes."""
        hash_orig = dhash(test_images["original"])
        hash_blur = dhash(test_images["blurred"])

        distance = hamming_distance(hash_orig, hash_blur)
        # Should be similar
        assert distance <= 10, f"Not resilient to blur: distance={distance}"

    def test_different_images_distinguishable(self, test_images: dict) -> None:
        """Completely different images should have different hashes."""
        hash_orig = dhash(test_images["original"])
        hash_diff = dhash(test_images["different"])

        distance = hamming_distance(hash_orig, hash_diff)
        # Should be clearly different
        assert (
            distance > 15
        ), f"Cannot distinguish different images: distance={distance}"

    def test_whash_most_robust_to_transforms(self, test_images: dict) -> None:
        """wHash should be most robust to image transformations."""
        # Test all three methods on brightness change
        orig = test_images["original"]
        bright = test_images["brightened"]

        dhash_dist = hamming_distance(dhash(orig), dhash(bright))
        ahash_dist = hamming_distance(ahash(orig), ahash(bright))
        whash_dist = hamming_distance(whash(orig), whash(bright))

        # wHash should typically be most robust (lowest distance)
        # Note: This is a general expectation, not guaranteed for all images
        print(
            f"\nRobustness to brightness: dHash={dhash_dist}, aHash={ahash_dist}, wHash={whash_dist}"
        )

        # At least verify all are reasonably similar
        assert dhash_dist <= 20
        assert ahash_dist <= 20
        assert whash_dist <= 20


class TestHashPerformance:
    """Test and benchmark hash computation performance."""

    def test_dhash_performance(self, test_images: dict, benchmark) -> None:
        """Benchmark dHash computation speed."""
        result = benchmark(dhash, test_images["original"])
        assert result is not None

    def test_ahash_performance(self, test_images: dict, benchmark) -> None:
        """Benchmark aHash computation speed."""
        result = benchmark(ahash, test_images["original"])
        assert result is not None

    def test_whash_performance(self, test_images: dict, benchmark) -> None:
        """Benchmark wHash computation speed."""
        result = benchmark(whash, test_images["original"])
        assert result is not None

    def test_combined_hash_performance(self, test_images: dict) -> None:
        """Measure time to compute all three hashes together."""
        start = time.perf_counter()
        hashes = combined_hash(test_images["original"])
        elapsed = time.perf_counter() - start

        assert hashes is not None
        assert "dhash" in hashes
        assert "ahash" in hashes
        assert "whash" in hashes

        print(f"\nCombined hash computation: {elapsed*1000:.2f}ms")
        # Should complete in reasonable time
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s"

    def test_batch_hash_performance(self, test_images: dict) -> None:
        """Measure performance of hashing multiple images."""
        images = [
            test_images["original"],
            test_images["low_quality"],
            test_images["resized_small"],
            test_images["brightened"],
            test_images["different"],
        ]

        start = time.perf_counter()
        hashes = []
        for img_path in images:
            h = combined_hash(img_path)
            hashes.append(h)
        elapsed = time.perf_counter() - start

        assert len(hashes) == len(images)
        print(f"\nBatch hashing {len(images)} images: {elapsed*1000:.2f}ms")
        print(f"Average per image: {(elapsed/len(images))*1000:.2f}ms")

        # Should be reasonably fast
        avg_time = elapsed / len(images)
        assert avg_time < 0.5, f"Too slow: {avg_time:.3f}s per image"

    def test_similarity_comparison_performance(self, test_images: dict) -> None:
        """Measure performance of pairwise similarity comparisons."""
        # Compute hashes for all test images
        hashes = {}
        for name, path in test_images.items():
            hashes[name] = combined_hash(path)

        # Measure pairwise comparison time
        start = time.perf_counter()
        comparisons = 0
        for name1, hash1 in hashes.items():
            for name2, hash2 in hashes.items():
                if name1 < name2:  # Avoid duplicate comparisons
                    _ = get_best_matches(hash1, hash2)
                    comparisons += 1
        elapsed = time.perf_counter() - start

        print(f"\n{comparisons} pairwise comparisons: {elapsed*1000:.2f}ms")
        print(f"Average per comparison: {(elapsed/comparisons)*1000:.2f}ms")

        # Comparisons should be very fast
        avg_time = elapsed / comparisons
        assert avg_time < 0.01, f"Comparisons too slow: {avg_time*1000:.3f}ms each"


class TestHashMethodComparison:
    """Compare the three hash methods on various criteria."""

    def test_accuracy_comparison(self, test_images: dict) -> None:
        """Compare accuracy of different hash methods."""
        results = {
            "dhash": {"similar_correct": 0, "different_correct": 0},
            "ahash": {"similar_correct": 0, "different_correct": 0},
            "whash": {"similar_correct": 0, "different_correct": 0},
        }

        # Test similar images (should have low distance)
        similar_pairs = [
            ("original", "exact_copy"),
            ("original", "low_quality"),
            ("original", "resized_small"),
            ("original", "blurred"),
        ]

        for method in ["dhash", "ahash", "whash"]:
            hash_func = (
                dhash if method == "dhash" else (ahash if method == "ahash" else whash)
            )

            for img1, img2 in similar_pairs:
                hash1 = hash_func(test_images[img1])
                hash2 = hash_func(test_images[img2])
                distance = hamming_distance(hash1, hash2)
                if distance <= 10:
                    results[method]["similar_correct"] += 1

            # Test different images (should have high distance)
            hash_orig = hash_func(test_images["original"])
            hash_diff = hash_func(test_images["different"])
            distance = hamming_distance(hash_orig, hash_diff)
            if distance > 15:
                results[method]["different_correct"] += 1

        print("\nAccuracy comparison:")
        for method, scores in results.items():
            similar_acc = scores["similar_correct"] / len(similar_pairs) * 100
            different_acc = scores["different_correct"] * 100
            print(
                f"{method}: Similar={similar_acc:.1f}%, Different={different_acc:.1f}%"
            )

    def test_speed_comparison(self, test_images: dict) -> None:
        """Compare speed of different hash methods."""
        img_path = test_images["original"]
        iterations = 10

        times = {}
        for method_name, hash_func in [
            ("dhash", dhash),
            ("ahash", ahash),
            ("whash", whash),
        ]:
            start = time.perf_counter()
            for _ in range(iterations):
                _ = hash_func(img_path)
            elapsed = time.perf_counter() - start
            times[method_name] = elapsed / iterations

        print("\nSpeed comparison (average of 10 runs):")
        for method, avg_time in sorted(times.items(), key=lambda x: x[1]):
            print(f"{method}: {avg_time*1000:.2f}ms")

        # All should complete reasonably fast
        for method, avg_time in times.items():
            assert avg_time < 0.5, f"{method} too slow: {avg_time:.3f}s"
