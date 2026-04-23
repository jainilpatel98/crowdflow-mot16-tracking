# Model Readiness Status

This file answers one question clearly:

## Which model is fully ready right now?

The current fully ready class-demo pipeline is:

- detector: `yolo26n.pt`
- tracker: `BoT-SORT`
- tracker config: `code/botsort_mot16_person.yaml`

This pipeline is ready because:

- it has a completed held-out report on `MOT16-11` and `MOT16-13`
- it has saved preview videos
- it has saved MOT-format tracking files
- it passes the automated validation suite

Best completed report currently available:

- `tracking_eval/runs/20260413_005829_final_val_yolo26n_botsort_960/report.md`

Overall metrics from that report:

- `MOTA = 0.4040`
- `IDF1 = 0.4971`
- `Precision = 0.8978`
- `Recall = 0.4656`

## Is this detector custom-trained in this repo?

No.

The current ready detector appears to be a pretrained YOLO checkpoint, not a repo-trained checkpoint.
The loaded checkpoint metadata points to COCO-style training settings rather than a project-local MOT16 fine-tuning run.

That is acceptable for the class project if you present it honestly as the detector used in the final pipeline.

## Is the Faster R-CNN baseline fully trained right now?

Not yet, based on the current repo state.

What I checked:

- `output/fasterrcnn/` currently has no saved checkpoint files
- `output/fasterrcnn_tracking_results/` currently has no saved tracking results
- `output/fasterrcnn_videos/` currently has no saved exported videos

The notebook contains a training plan for:

- 10 epochs
- checkpoint path `output/fasterrcnn/fasterrcnn_heads_e10_best.pth`
- validation sequences `MOT16-11` and `MOT16-13`

But until that checkpoint and the downstream outputs exist, this baseline should be treated as:

- prepared training pipeline
- not yet finished training result

## When can we say the custom-trained model is ready?

Only after all of the following are true:

1. `output/fasterrcnn/fasterrcnn_heads_e10_best.pth` exists.
2. Inference on a validation sequence completes.
3. MOT-format tracking output is exported.
4. A preview video or GIF is exported.
5. Validation metrics are collected and saved.
6. The project validation suite still passes after any code changes.

## Recommendation for class

Use the `yolo26n + BoT-SORT` pipeline as the main demo and main quantitative report result.

If you still want to include the Faster R-CNN notebook:

- present it as a custom-training baseline path
- only call it completed if the checkpoint and evaluation artifacts are actually created before class
