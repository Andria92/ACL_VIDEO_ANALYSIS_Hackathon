from acl_motion.video.roi import BBox


def test_bbox_from_string_and_clamp():
    bbox = BBox.from_string("10,20,100,200")

    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 100
    assert bbox.height == 200

    clamped = bbox.clamp(image_width=80, image_height=120)

    assert clamped.x == 10
    assert clamped.y == 20
    assert clamped.width == 70
    assert clamped.height == 100


def test_bbox_padding_preserves_center():
    bbox = BBox(10, 20, 100, 50)

    padded = bbox.pad(0.1)

    assert padded.x == 0
    assert padded.y == 15
    assert padded.width == 120
    assert padded.height == 60


def test_bbox_rejects_non_positive_size():
    try:
        BBox(0, 0, 0, 10)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected BBox to reject zero width.")
