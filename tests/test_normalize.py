from datetime import date

from observatory.sources.base import normalize_tiktok_item

ITEM = {
    "id": 123,
    "desc": "cat flies a plane #AIvideo #Sora",
    "createTime": 1760000000,
    "author": {"uniqueId": "someone"},
    "music": {"id": 9},
    "challenges": [],
    "stats": {"playCount": 1000, "diggCount": 50, "commentCount": 3, "shareCount": 7},
}


def test_maps_tiktok_shape():
    snap = normalize_tiktok_item(ITEM, date(2026, 10, 1), "fixture")
    assert snap.video_id == "123"
    assert snap.views == 1000 and snap.likes == 50
    assert snap.author == "someone"
    assert snap.created_at.year == 2025


def test_hashtags_fall_back_to_caption_and_are_normalized():
    snap = normalize_tiktok_item(ITEM, date(2026, 10, 1), "fixture")
    assert snap.hashtags == ["aivideo", "sora"]
