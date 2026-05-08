from pathlib import Path

from PIL import Image

from reid_tools.mot16_reid_dataset import read_mot16_reid_samples


def test_reid_samples_keep_person_like_classes_only(tmp_path: Path):
    seq_dir = tmp_path / "train" / "MOT16-99"
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100)).save(img_dir / "000001.jpg")
    (gt_dir / "gt.txt").write_text(
        "\n".join(
            [
                "1,1,10,10,20,30,1,1,0.9",
                "1,2,10,10,20,30,1,2,0.9",
                "1,3,10,10,20,30,1,3,0.9",
                "1,4,10,10,20,30,0,1,0.9",
                "1,5,10,10,20,30,1,1,0.1",
            ]
        ),
        encoding="utf-8",
    )

    samples = read_mot16_reid_samples(root=tmp_path, sequences=["MOT16-99"], class_ids={1, 2}, min_visibility=0.25)

    assert [sample.track_id for sample in samples] == [1, 2]
    assert {sample.class_id for sample in samples} == {1, 2}
