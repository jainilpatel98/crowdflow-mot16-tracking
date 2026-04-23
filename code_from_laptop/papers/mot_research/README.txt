MOT RESEARCH PAPER PACK

Created: 2026-03-23

Purpose:
Small curated paper set for the MOT16 project, focused on benchmark understanding, identity preservation, re-identification, and modern tracker design.


Downloaded files

1. MOT16_A_Benchmark_for_Multi_Object_Tracking.pdf
- Benchmark paper for MOT16
- Use for dataset, protocol, and evaluation citations

2. Deep_SORT_A_Deep_Association_Metric.pdf
- Classic appearance-aware online tracking paper
- Best starting point for explaining why Re-ID helps reduce ID switches

3. Hierarchical_Deep_Tracklet_ReIdentification.pdf
- Tracklet-level re-identification paper
- Useful for long-occlusion and ID reactivation discussion

4. OSNet_Omni_Scale_Feature_Learning_for_Person_ReID.pdf
- Practical person Re-ID backbone paper
- Useful if the project adds learned appearance embeddings

5. ByteTrack_Associating_Every_Detection_Box.pdf
- Important modern tracking paper
- Useful for understanding low-confidence detection recovery

6. BoT_SORT_Robust_Associations_Multi_Pedestrian_Tracking.pdf
- Strong MOT tracker combining motion, appearance, and camera-motion compensation
- Most relevant paper for the YOLO + BoT-SORT notebook path

7. FeatureSORT_Essential_Features_for_Effective_Tracking.pdf
- Recent MOT16-leading paper
- Especially relevant because it combines color, style, direction, and Re-ID instead of relying on color alone

8. FastTracker_Real_Time_and_Accurate_Visual_Tracking.pdf
- Recent MOT16-leading paper on the benchmark page
- Useful for occlusion-aware re-identification ideas

9. In_Defense_of_the_Triplet_Loss_for_Person_ReID.pdf
- Metric-learning reference for person Re-ID
- Useful if the report discusses Siamese or embedding-based identity matching


Suggested reading order

Read first:

- MOT16_A_Benchmark_for_Multi_Object_Tracking.pdf
- Deep_SORT_A_Deep_Association_Metric.pdf
- ByteTrack_Associating_Every_Detection_Box.pdf
- BoT_SORT_Robust_Associations_Multi_Pedestrian_Tracking.pdf

Read next:

- OSNet_Omni_Scale_Feature_Learning_for_Person_ReID.pdf
- FeatureSORT_Essential_Features_for_Effective_Tracking.pdf
- Hierarchical_Deep_Tracklet_ReIdentification.pdf

Read if needed:

- FastTracker_Real_Time_and_Accurate_Visual_Tracking.pdf
- In_Defense_of_the_Triplet_Loss_for_Person_ReID.pdf


Most useful papers for this exact project

If the project only has time to rely on three outside papers, use:

- MOT16_A_Benchmark_for_Multi_Object_Tracking.pdf
- Deep_SORT_A_Deep_Association_Metric.pdf
- BoT_SORT_Robust_Associations_Multi_Pedestrian_Tracking.pdf

If the project specifically emphasizes ID reassignment or Siamese / Re-ID:

- Deep_SORT_A_Deep_Association_Metric.pdf
- OSNet_Omni_Scale_Feature_Learning_for_Person_ReID.pdf
- In_Defense_of_the_Triplet_Loss_for_Person_ReID.pdf
- Hierarchical_Deep_Tracklet_ReIdentification.pdf

