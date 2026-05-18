import hashlib


def split_for_id(sample_id: str) -> str:
    bucket = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"
