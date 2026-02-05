# Experiments Log

## 2026-02-05
- Model: LightGCN baseline + greedy genre-diversity rerank (top-20)
- Notes: first end-to-end submission from initial LightGCN notebook (3 epochs, 300 steps/epoch)
- Public LB score: 0.06341180784089523
- Update: same setup with longer training (15 epochs, 500 steps/epoch)
- Public LB score: 0.0706

- Baseline: Random candidate ordering + greedy genre-diversity rerank (top-20)
- Public LB score: 0.1877

- Baseline: Popularity + greedy genre-diversity rerank (top-20)
- Local NDCG@20: 0.017486
- Local Diversity@20: 0.012285
- Local Score: 0.015926
- Public LB score: 0.03436976728490539

- Baseline: Recency Popularity (tau=60d) + greedy genre-diversity rerank (top-20)
- Local NDCG@20: 0.020657
- Local Diversity@20: 0.013433
- Local Score: 0.018490
- Public LB score: 0.0687

- Baseline: User Genre Profile + greedy genre-diversity rerank (top-20)
- Local NDCG@20: 0.161823
- Local Diversity@20: 0.141498
- Local Score: 0.155725
- Public LB score: 0.1456

- Baseline: Co-Read + greedy genre-diversity rerank (top-20)
- Local NDCG@20: 0.048017
- Local Diversity@20: 0.058133
- Local Score: 0.051052
- Public LB score: 0.0687
