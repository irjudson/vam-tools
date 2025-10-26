"""
Tests for quality scoring module.
"""

import pytest

from vam_tools.v2.analysis.quality_scorer import (
    _score_file_size,
    _score_format,
    _score_metadata_completeness,
    _score_resolution,
    calculate_quality_score,
    compare_quality,
    select_best,
)
from vam_tools.v2.core.types import FileType, ImageMetadata


class TestScoreFormat:
    """Tests for format scoring."""

    def test_score_raw_formats(self) -> None:
        """Test that RAW formats get highest scores."""
        assert _score_format("CR2") == 100.0
        assert _score_format("NEF") == 100.0
        assert _score_format("ARW") == 100.0
        assert _score_format("DNG") == 100.0

    def test_score_lossless_formats(self) -> None:
        """Test that lossless formats get high scores."""
        assert _score_format("TIFF") == 90.0
        assert _score_format("PNG") == 85.0

    def test_score_lossy_formats(self) -> None:
        """Test that lossy formats get lower scores."""
        assert _score_format("JPEG") == 70.0
        assert _score_format("WEBP") == 60.0

    def test_score_case_insensitive(self) -> None:
        """Test that format scoring is case-insensitive."""
        assert _score_format("jpeg") == _score_format("JPEG")
        assert _score_format("Png") == _score_format("png")

    def test_score_unknown_format(self) -> None:
        """Test that unknown formats get neutral score."""
        assert _score_format("XYZ") == 50.0
        assert _score_format("") == 50.0


class TestScoreResolution:
    """Tests for resolution scoring."""

    def test_score_8k_resolution(self) -> None:
        """Test 8K resolution gets max score."""
        score = _score_resolution(7680, 4320)  # 8K
        assert score == 100.0

    def test_score_4k_resolution(self) -> None:
        """Test 4K resolution gets high score."""
        score = _score_resolution(3840, 2160)  # 4K
        assert 80 <= score <= 90

    def test_score_1080p_resolution(self) -> None:
        """Test 1080p resolution gets medium-high score."""
        score = _score_resolution(1920, 1080)  # 1080p
        assert 60 <= score <= 80

    def test_score_720p_resolution(self) -> None:
        """Test 720p resolution gets medium score."""
        score = _score_resolution(1280, 720)  # 720p
        assert 35 <= score <= 50

    def test_score_very_low_resolution(self) -> None:
        """Test very low resolution gets low score."""
        score = _score_resolution(320, 240)
        assert score < 40

    def test_score_invalid_resolution(self) -> None:
        """Test invalid resolution returns 0."""
        assert _score_resolution(None, 1080) == 0.0
        assert _score_resolution(1920, None) == 0.0
        assert _score_resolution(0, 0) == 0.0
        assert _score_resolution(-100, 100) == 0.0

    def test_score_resolution_scales(self) -> None:
        """Test that higher resolution always scores higher."""
        score1 = _score_resolution(1920, 1080)
        score2 = _score_resolution(3840, 2160)
        score3 = _score_resolution(7680, 4320)

        assert score1 < score2 < score3


class TestScoreFileSize:
    """Tests for file size scoring."""

    def test_score_raw_file_sizes(self) -> None:
        """Test RAW file size scoring."""
        # Small RAW
        score_small = _score_file_size(10 * 1024 * 1024, "CR2")  # 10 MB
        # Medium RAW
        score_medium = _score_file_size(30 * 1024 * 1024, "NEF")  # 30 MB
        # Large RAW
        score_large = _score_file_size(60 * 1024 * 1024, "ARW")  # 60 MB

        assert score_small < score_medium < score_large
        assert score_large == 100.0

    def test_score_jpeg_file_sizes(self) -> None:
        """Test JPEG file size scoring."""
        # Tiny JPEG
        score_tiny = _score_file_size(500 * 1024, "JPEG")  # 500 KB
        # Medium JPEG
        score_medium = _score_file_size(5 * 1024 * 1024, "JPEG")  # 5 MB
        # Large JPEG
        score_large = _score_file_size(15 * 1024 * 1024, "JPG")  # 15 MB

        assert score_tiny < score_medium < score_large

    def test_score_invalid_file_size(self) -> None:
        """Test invalid file sizes return 0."""
        assert _score_file_size(None, "JPEG") == 0.0
        assert _score_file_size(0, "PNG") == 0.0
        assert _score_file_size(-1000, "TIFF") == 0.0


class TestScoreMetadataCompleteness:
    """Tests for metadata completeness scoring."""

    def test_score_complete_metadata(self) -> None:
        """Test that complete metadata gets high score."""
        metadata = ImageMetadata(
            camera_make="Canon",
            camera_model="EOS 5D Mark IV",
            lens_model="EF 24-70mm f/2.8L II USM",
            focal_length=50.0,
            aperture=2.8,
            shutter_speed="1/500",
            iso=400,
            gps_latitude=37.7749,
            gps_longitude=-122.4194,
        )

        score = _score_metadata_completeness(metadata)
        assert score == 100.0

    def test_score_minimal_metadata(self) -> None:
        """Test that minimal metadata gets low score."""
        metadata = ImageMetadata()
        score = _score_metadata_completeness(metadata)
        assert score == 0.0

    def test_score_partial_metadata(self) -> None:
        """Test that partial metadata gets partial score."""
        metadata = ImageMetadata(
            camera_make="Nikon",
            camera_model="D850",
            iso=200,
        )

        score = _score_metadata_completeness(metadata)
        assert 0 < score < 100
        assert 30 <= score <= 50  # Should have about 40% of fields

    def test_score_camera_only_metadata(self) -> None:
        """Test metadata with only camera info."""
        metadata = ImageMetadata(
            camera_make="Sony",
            camera_model="A7R IV",
        )

        score = _score_metadata_completeness(metadata)
        assert 20 <= score <= 40  # Camera make + model = 30%


class TestCalculateQualityScore:
    """Tests for overall quality score calculation."""

    def test_calculate_quality_high_quality_image(self) -> None:
        """Test quality score for high quality image."""
        metadata = ImageMetadata(
            format="CR2",  # RAW
            width=6720,
            height=4480,
            size_bytes=30 * 1024 * 1024,  # 30 MB
            camera_make="Canon",
            camera_model="EOS 5D Mark IV",
            lens_model="EF 24-70mm f/2.8L",
            focal_length=50.0,
            aperture=2.8,
            shutter_speed="1/500",
            iso=400,
        )

        score = calculate_quality_score(metadata, FileType.IMAGE)

        assert score.overall > 90  # Should be very high
        assert score.format_score == 100.0
        assert score.resolution_score > 90
        assert score.metadata_score > 60

    def test_calculate_quality_low_quality_image(self) -> None:
        """Test quality score for low quality image."""
        metadata = ImageMetadata(
            format="GIF",
            width=320,
            height=240,
            size_bytes=100 * 1024,  # 100 KB
        )

        score = calculate_quality_score(metadata, FileType.IMAGE)

        assert score.overall < 50  # Should be low
        assert score.format_score == 40.0  # GIF
        assert score.resolution_score < 40
        assert score.metadata_score == 0

    def test_calculate_quality_medium_quality_jpeg(self) -> None:
        """Test quality score for medium quality JPEG."""
        metadata = ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=3 * 1024 * 1024,  # 3 MB
            camera_make="Apple",
            camera_model="iPhone 12",
        )

        score = calculate_quality_score(metadata, FileType.IMAGE)

        assert 50 < score.overall < 80  # Medium quality
        assert score.format_score == 70.0  # JPEG
        assert 60 < score.resolution_score < 80  # 1080p

    def test_score_components_within_range(self) -> None:
        """Test that all score components are between 0 and 100."""
        metadata = ImageMetadata(
            format="PNG",
            width=2048,
            height=1536,
            size_bytes=5 * 1024 * 1024,
        )

        score = calculate_quality_score(metadata, FileType.IMAGE)

        assert 0 <= score.overall <= 100
        assert 0 <= score.format_score <= 100
        assert 0 <= score.resolution_score <= 100
        assert 0 <= score.size_score <= 100
        assert 0 <= score.metadata_score <= 100


class TestCompareQuality:
    """Tests for quality comparison."""

    def test_compare_raw_vs_jpeg(self) -> None:
        """Test that RAW is preferred over JPEG."""
        raw_metadata = ImageMetadata(
            format="CR2",
            width=6000,
            height=4000,
            size_bytes=25 * 1024 * 1024,
        )

        jpeg_metadata = ImageMetadata(
            format="JPEG",
            width=6000,
            height=4000,
            size_bytes=5 * 1024 * 1024,
        )

        result = compare_quality(
            raw_metadata, FileType.IMAGE, jpeg_metadata, FileType.IMAGE
        )

        assert result == -1  # RAW is better

    def test_compare_high_res_vs_low_res(self) -> None:
        """Test that higher resolution is preferred."""
        high_res = ImageMetadata(
            format="JPEG",
            width=4000,
            height=3000,
            size_bytes=8 * 1024 * 1024,
        )

        low_res = ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=2 * 1024 * 1024,
        )

        result = compare_quality(high_res, FileType.IMAGE, low_res, FileType.IMAGE)

        assert result == -1  # High res is better

    def test_compare_identical_quality(self) -> None:
        """Test comparison of identical quality images."""
        metadata1 = ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=3 * 1024 * 1024,
        )

        metadata2 = ImageMetadata(
            format="JPEG",
            width=1920,
            height=1080,
            size_bytes=3 * 1024 * 1024,
        )

        result = compare_quality(metadata1, FileType.IMAGE, metadata2, FileType.IMAGE)

        assert result == 0  # Equal quality


class TestSelectBest:
    """Tests for selecting best image from a group."""

    def test_select_best_from_group(self) -> None:
        """Test selecting best image from diverse group."""
        images = {
            "img1": (
                ImageMetadata(
                    format="JPEG", width=1920, height=1080, size_bytes=2 * 1024 * 1024
                ),
                FileType.IMAGE,
            ),
            "img2": (
                ImageMetadata(
                    format="CR2", width=6000, height=4000, size_bytes=25 * 1024 * 1024
                ),
                FileType.IMAGE,
            ),
            "img3": (
                ImageMetadata(
                    format="PNG", width=2048, height=1536, size_bytes=8 * 1024 * 1024
                ),
                FileType.IMAGE,
            ),
        }

        best_id, best_score = select_best(images)

        assert best_id == "img2"  # RAW file should be best
        assert best_score.overall > 85

    def test_select_best_single_image(self) -> None:
        """Test selecting from single image."""
        images = {
            "only_img": (
                ImageMetadata(
                    format="JPEG", width=1920, height=1080, size_bytes=3 * 1024 * 1024
                ),
                FileType.IMAGE,
            ),
        }

        best_id, best_score = select_best(images)

        assert best_id == "only_img"
        assert best_score.overall > 0

    def test_select_best_empty_raises(self) -> None:
        """Test that empty image dict raises ValueError."""
        with pytest.raises(ValueError):
            select_best({})

    def test_select_best_prefers_complete_metadata(self) -> None:
        """Test that complete metadata breaks ties."""
        images = {
            "minimal": (
                ImageMetadata(
                    format="JPEG", width=1920, height=1080, size_bytes=3 * 1024 * 1024
                ),
                FileType.IMAGE,
            ),
            "complete": (
                ImageMetadata(
                    format="JPEG",
                    width=1920,
                    height=1080,
                    size_bytes=3 * 1024 * 1024,
                    camera_make="Canon",
                    camera_model="EOS R5",
                    lens_model="RF 24-70mm",
                    focal_length=50.0,
                    aperture=2.8,
                    shutter_speed="1/500",
                    iso=400,
                ),
                FileType.IMAGE,
            ),
        }

        best_id, _ = select_best(images)

        assert best_id == "complete"  # More metadata wins
