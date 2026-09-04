# SciFact Failure Analysis

## Methodology

- dataset: scifact (BeIR/scifact)
- queries evaluated: 300 (all with qrels)
- methods: BM25, Dense, Hybrid Weighted, Hybrid + rerank@20
- evaluation k = 10; hybrid alpha = 0.5; hybrid candidate_k = 50; reranker pool = 20
- dense model: sentence-transformers/all-MiniLM-L6-v2
- reranker model: cross-encoder/ms-marco-MiniLM-L-6-v2
- primary comparison metric: nDCG@10 (MRR@10 / Recall@10 as diagnostics)
- category definitions:
  - bm25_wins: BM25 nDCG@10 minus Dense nDCG@10 > 0; ranked by largest difference
  - dense_wins: Dense nDCG@10 minus BM25 nDCG@10 > 0; ranked by largest difference
  - hybrid_wins: Hybrid Weighted nDCG@10 > max(BM25, Dense) nDCG@10; ranked by largest margin
  - reranker_improves: Reranked nDCG@10 minus Hybrid Weighted nDCG@10 > 0; ranked by largest gain
  - reranker_hurts: Hybrid Weighted nDCG@10 minus Reranked nDCG@10 > 0; ranked by largest loss
  - all_fail: Recall@10 == 0 for BM25, Dense, Hybrid Weighted, and Hybrid + rerank@20; ordered by ascending query id
- selection: up to 3 examples per category, ranked by category score with ascending query id as the deterministic tie-break

## Summary

Category counts (qualifying queries):

| Category | Qualifying queries |
|---|---|
| BM25 Wins, Dense Loses | 54 |
| Dense Wins, BM25 Loses | 92 |
| Hybrid Wins | 18 |
| Reranker Improves | 62 |
| Reranker Hurts | 49 |
| All Methods Fail | 37 |

Mean metrics per method:

| Method | precision@10 | recall@10 | mrr@10 | ndcg@10 |
|---|---|---:|---:|---:|
| BM25 | 0.0760 | 0.6803 | 0.5381 | 0.5694 |
| Dense | 0.0883 | 0.7833 | 0.6047 | 0.6451 |
| Hybrid Weighted | 0.0907 | 0.8033 | 0.6413 | 0.6771 |
| Hybrid + rerank@20 | 0.0937 | 0.8356 | 0.6664 | 0.6998 |

## BM25 Wins, Dense Loses

Qualifying queries: 54; shown: 3.

### Query 1024: Recurrent mutations occur frequently within CTCF anchor sites adjacent to oncogenes.

Relevant document(s):
- 5373138 (grade 1): 3D Chromosome Regulatory Landscape of Human Pluripotent Cells. In this study, we describe the 3D chromosome regulatory landscape of human naive and primed embryonic stem cells. To devise this map, we... — exact query-token overlap 6 (ratio 0.55)

Per-method top-10:
- BM25: [5373138, 24995939, 140874, 8494570, 14530534, 14973286, 24896957, 4312169, 10650521, 38630735] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['5373138']
- Dense: [8494570, 36713289, 4926049, 6078882, 4457160, 6532806, 667451, 39462488, 7468449, 24995939] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [8494570, 24995939, 5373138, 24896957, 140874, 36713289, 6078882, 4926049, 14530534, 4457160] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.3333, ndcg@10=0.5000 — first relevant rank 3 — relevant in top-10: ['5373138']
- Hybrid + rerank@20: [5373138, 24995939, 140874, 8494570, 6078882, 14530534, 24896957, 14973286, 37249641, 36082224] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['5373138']

Observation:
  BM25 nDCG@10 is 1.0000 vs Dense 0.0000.
  BM25 ranks the first relevant document #1 while Dense ranks it not in top-10.

Hypothesis:
  The query and relevant document 5373138 share 6 exact query token(s) (ratio 0.55); overlap may favor lexical retrieval, but overlap alone does not prove causation.

### Query 70: Activation of PPM1D suppresses p53 function.

Relevant document(s):
- 4414547 (grade 1): Mosaic PPM1D mutations are associated with predisposition to breast and ovarian cancer Improved sequencing technologies offer unprecedented opportunities for investigating the role of rare genetic var... — exact query-token overlap 3 (ratio 0.50)
- 5956380 (grade 1): Exome sequencing identifies somatic gain-of-function PPM1D mutations in brainstem gliomas Gliomas arising in the brainstem and thalamus are devastating tumors that are difficult to surgically resect.... — exact query-token overlap 4 (ratio 0.67)

Per-method top-10:
- BM25: [5956380, 4414547, 24721866, 23985464, 7225911, 4389394, 4460880, 43880096, 25682129, 11903247] — precision@10=0.2000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['5956380', '4414547']
- Dense: [9483851, 27949347, 11903247, 6896063, 28107602, 31624828, 9169645, 21557055, 37204802, 14460402] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [5956380, 9483851, 11903247, 4414547, 6896063, 9169645, 27949347, 4389394, 43880096, 21307488] — precision@10=0.2000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=0.8772 — first relevant rank 1 — relevant in top-10: ['5956380', '4414547']
- Hybrid + rerank@20: [21307488, 6896063, 21557055, 31624828, 21521236, 11903247, 9483851, 4389394, 24721866, 5254463] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []

Observation:
  BM25 nDCG@10 is 1.0000 vs Dense 0.0000.
  BM25 ranks the first relevant document #1 while Dense ranks it not in top-10.

Hypothesis:
  The query and relevant document 4414547 share 3 exact query token(s) (ratio 0.50); overlap may favor lexical retrieval, but overlap alone does not prove causation.

### Query 743: Macrolides protect against myocardial infarction.

Relevant document(s):
- 32159283 (grade 1): Antibiotics and risk of subsequent first-time acute myocardial infarction. CONTEXT Increasing evidence supports the hypothesis of a causal association between certain bacterial infections and increase... — exact query-token overlap 4 (ratio 0.80)

Per-method top-10:
- BM25: [32159283, 5884524, 5993745, 44264297, 34054472, 21239672, 43629704, 5085118, 2248870, 15194125] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['32159283']
- Dense: [3984231, 6270720, 7662206, 17897801, 4020950, 7873737, 6853699, 13923069, 2842550, 10576136] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [3984231, 32159283, 6270720, 17897801, 7662206, 4020950, 7873737, 1905095, 13923069, 6853699] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.5000, ndcg@10=0.6309 — first relevant rank 2 — relevant in top-10: ['32159283']
- Hybrid + rerank@20: [32159283, 3984231, 10576136, 6853699, 1905095, 17897801, 5884524, 8290953, 4020950, 25028913] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['32159283']

Observation:
  BM25 nDCG@10 is 1.0000 vs Dense 0.0000.
  BM25 ranks the first relevant document #1 while Dense ranks it not in top-10.

Hypothesis:
  The query and relevant document 32159283 share 4 exact query token(s) (ratio 0.80); overlap may favor lexical retrieval, but overlap alone does not prove causation.

## Dense Wins, BM25 Loses

Qualifying queries: 92; shown: 3.

### Query 1088: Silencing of Bcl2 is important for the maintenance and progression of tumors.

Relevant document(s):
- 37549932 (grade 1): Antiapoptotic BCL-2 is required for maintenance of a model leukemia. Resistance to apoptosis, often achieved by the overexpression of antiapoptotic proteins, is common and perhaps required in the gene... — exact query-token overlap 6 (ratio 0.55)

Per-method top-10:
- BM25: [14863011, 5765455, 20333864, 9539753, 3840043, 35443524, 21551568, 4462155, 1782201, 2613813] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [37549932, 4323425, 8496132, 11900630, 34439544, 16550075, 14863011, 19510470, 21793890, 33638477] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['37549932']
- Hybrid Weighted: [14863011, 37549932, 5765455, 4323425, 8496132, 11900630, 20333864, 34439544, 16550075, 3840043] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.5000, ndcg@10=0.6309 — first relevant rank 2 — relevant in top-10: ['37549932']
- Hybrid + rerank@20: [37549932, 33638477, 11900630, 19510470, 8496132, 34439544, 14863011, 4323425, 16550075, 5765455] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['37549932']

Observation:
  Dense nDCG@10 is 1.0000 vs BM25 0.0000.
  Dense ranks the first relevant document #1 while BM25 ranks it not in top-10.

Hypothesis:
  The query and relevant document 37549932 share 6 exact query token(s) (ratio 0.55); overlap may favor lexical retrieval, but overlap alone does not prove causation.

### Query 1137: TNFAIP3 is a tumor suppressor in glioblastoma.

Relevant document(s):
- 33370 (grade 1): Targeting A20 Decreases Glioma Stem Cell Survival and Tumor Growth Glioblastomas are deadly cancers that display a functional cellular hierarchy maintained by self-renewing glioblastoma stem cells (GS... — exact query-token overlap 4 (ratio 0.57)

Per-method top-10:
- BM25: [13244602, 9486930, 25726838, 29366489, 41293601, 19541444, 11578459, 23934390, 470625, 26735905] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [33370, 470625, 9486930, 5123516, 1836154, 37686718, 12685434, 38355793, 9737083, 14180565] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['33370']
- Hybrid Weighted: [9486930, 470625, 13244602, 33370, 12948892, 25726838, 29366489, 5123516, 41293601, 7898952] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.2500, ndcg@10=0.4307 — first relevant rank 4 — relevant in top-10: ['33370']
- Hybrid + rerank@20: [33370, 470625, 5123516, 7898952, 38355793, 12685434, 29366489, 13244602, 19541444, 9486930] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['33370']

Observation:
  Dense nDCG@10 is 1.0000 vs BM25 0.0000.
  Dense ranks the first relevant document #1 while BM25 ranks it not in top-10.

Hypothesis:
  The query and relevant document 33370 share 4 exact query token(s) (ratio 0.57); overlap may favor lexical retrieval, but overlap alone does not prove causation.

### Query 1194: The arm density of TatAd complexes is due to structural rearrangements within Class1 TatAd complexes such as the 'charge zipper mechanism'.

Relevant document(s):
- 11419230 (grade 1): Folding and Self-Assembly of the TatA Translocation Pore Based on a Charge Zipper Mechanism We propose a concept for the folding and self-assembly of the pore-forming TatA complex from the Twin-argini... — exact query-token overlap 5 (ratio 0.28)

Per-method top-10:
- BM25: [24706198, 22042345, 15327601, 37362689, 3400973, 22561064, 9164724, 13790144, 35256900, 14920021] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [11419230, 24706198, 22561064, 5473074, 16736883, 3400973, 12265561, 46248894, 6636088, 15659108] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['11419230']
- Hybrid Weighted: [24706198, 11419230, 22561064, 3400973, 15327601, 12265561, 16736883, 22042345, 5473074, 37362689] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.5000, ndcg@10=0.6309 — first relevant rank 2 — relevant in top-10: ['11419230']
- Hybrid + rerank@20: [11419230, 24706198, 22561064, 5473074, 22042345, 13992047, 16736883, 9164724, 6636088, 3400973] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['11419230']

Observation:
  Dense nDCG@10 is 1.0000 vs BM25 0.0000.
  Dense ranks the first relevant document #1 while BM25 ranks it not in top-10.

Hypothesis:
  The query and relevant document 11419230 share 5 exact query token(s) (ratio 0.28); overlap may favor lexical retrieval, but overlap alone does not prove causation.

## Hybrid Wins

Qualifying queries: 18; shown: 3.

### Query 637: Input from  mental and physical health care professionals is effective at decreasing homelessness.

Relevant document(s):
- 25649714 (grade 1): Mental health problems of homeless children and families: longitudinal study. OBJECTIVE To establish the mental health needs of homeless children and families before and after rehousing. DESIGN Cross... — exact query-token overlap 7 (ratio 0.54)

Per-method top-10:
- BM25: [73473433, 31612088, 6517267, 25649714, 20491205, 29845974, 19071857, 24988745, 24318630, 6647414] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.2500, ndcg@10=0.4307 — first relevant rank 4 — relevant in top-10: ['25649714']
- Dense: [31019903, 154243324, 33135135, 25649714, 6129301, 20334484, 31495049, 24318630, 15968271, 35022568] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.2500, ndcg@10=0.4307 — first relevant rank 4 — relevant in top-10: ['25649714']
- Hybrid Weighted: [25649714, 31019903, 73473433, 154243324, 6129301, 20491205, 31612088, 33135135, 24318630, 6517267] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['25649714']
- Hybrid + rerank@20: [6129301, 25649714, 31019903, 154243324, 20334484, 73473433, 33135135, 31612088, 20606520, 18872233] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.5000, ndcg@10=0.6309 — first relevant rank 2 — relevant in top-10: ['25649714']

Observation:
  Hybrid Weighted nDCG@10 is 1.0000, beating BM25 (0.4307) and Dense (0.4307).

Hypothesis:
  Fusing the rankings moves relevant document 25649714 to rank 1 (BM25: 4, Dense: 4); combining complementary evidence can promote documents neither retriever ranks first. Hypothesis only.

### Query 659: Ivermectin is used to treat lymphatic filariasis.

Relevant document(s):
- 1215116 (grade 1): “Rapid-Impact Interventions”: How a Policy of Integrated Control for Africa's Neglected Tropical Diseases Could Benefit the Poor Over the past two decades there have been significant achievements in t... — exact query-token overlap 3 (ratio 0.43)

Per-method top-10:
- BM25: [17327939, 46926352, 3896759, 13070316, 46765242, 22613657, 11481946, 27240699, 27567994, 29634262] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [18321590, 32454714, 46602807, 1215116, 2274272, 38043606, 14082855, 34481589, 28821565, 1386103] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.2500, ndcg@10=0.4307 — first relevant rank 4 — relevant in top-10: ['1215116']
- Hybrid Weighted: [1215116, 17327939, 18321590, 46926352, 32454714, 46602807, 3896759, 2274272, 38043606, 13070316] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['1215116']
- Hybrid + rerank@20: [46765242, 13070316, 20428155, 21258863, 34481589, 1215116, 17327939, 32454714, 46926352, 2947124] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.1667, ndcg@10=0.3562 — first relevant rank 6 — relevant in top-10: ['1215116']

Observation:
  Hybrid Weighted nDCG@10 is 1.0000, beating BM25 (0.0000) and Dense (0.4307).

Hypothesis:
  Fusing the rankings moves relevant document 1215116 to rank 1 (BM25: None, Dense: 4); combining complementary evidence can promote documents neither retriever ranks first. Hypothesis only.

### Query 839: Nanoparticles can be targeted against specific cell types by incorporating aptamers into lipid nanoparticles.

Relevant document(s):
- 1469751 (grade 1): Aptamer-functionalized lipid nanoparticles targeting osteoblasts as a novel RNA interference–based bone anabolic strategy Currently, major concerns about the safety and efficacy of RNA interference (R... — exact query-token overlap 4 (ratio 0.29)

Per-method top-10:
- BM25: [169264, 13923069, 1469751, 17327939, 6219790, 39465575, 2810997, 6936141, 33667484, 39048693] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.3333, ndcg@10=0.5000 — first relevant rank 3 — relevant in top-10: ['1469751']
- Dense: [6219790, 17327939, 1469751, 8182950, 4435369, 25435456, 13923069, 31715818, 13764090, 14692646] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.3333, ndcg@10=0.5000 — first relevant rank 3 — relevant in top-10: ['1469751']
- Hybrid Weighted: [1469751, 17327939, 6219790, 13923069, 169264, 8182950, 4435369, 25435456, 31715818, 13764090] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['1469751']
- Hybrid + rerank@20: [1469751, 17327939, 13923069, 6219790, 8182950, 13764090, 169264, 31715818, 20646904, 10982689] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['1469751']

Observation:
  Hybrid Weighted nDCG@10 is 1.0000, beating BM25 (0.5000) and Dense (0.5000).

Hypothesis:
  Fusing the rankings moves relevant document 1469751 to rank 1 (BM25: 3, Dense: 3); combining complementary evidence can promote documents neither retriever ranks first. Hypothesis only.

## Reranker Improves

Qualifying queries: 62; shown: 3.

### Query 1368: Vitamin D deficiency effects the term of delivery.

Relevant document(s):
- 2425364 (grade 1): Association between maternal serum 25-hydroxyvitamin D level and pregnancy and neonatal outcomes: systematic review and meta-analysis of observational studies. OBJECTIVE To assess the effect of 25-hyd... — exact query-token overlap 4 (ratio 0.50)

Per-method top-10:
- BM25: [13312471, 275294, 23267371, 21553394, 22023404, 12074066, 3150030, 8856690, 30720103, 16256507] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [9555784, 3150030, 36960449, 30720103, 23267371, 21553394, 3034412, 8856690, 38551172, 16256507] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [23267371, 13312471, 21553394, 3150030, 30720103, 36960449, 9555784, 8856690, 16256507, 22023404] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid + rerank@20: [2425364, 8856690, 23267371, 13312471, 9555784, 21553394, 22023404, 275294, 30720103, 3034412] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['2425364']

Observation:
  The reranker improves nDCG@10: Hybrid 0.0000 vs reranked 1.0000.
  Relevant document 2425364 moves from hybrid-top-20 rank 19 to final rank 1.

Hypothesis:
  The cross-encoder's pairwise scores disagree with the hybrid fusion ranking in this case, which may explain the reordering; hypothesis only.

### Query 690: Less than 10% of the gabonese children with Schimmelpenning-Feuerstein-Mims syndrome (SFM) had a plasma lactate of more than 5mmol/L.

Relevant document(s):
- 18750453 (grade 1): Assessment of Volume Depletion in Children with Malaria Background The degree of volume depletion in severe malaria is currently unknown, although knowledge of fluid compartment volumes can guide ther... — exact query-token overlap 7 (ratio 0.41)

Per-method top-10:
- BM25: [23908217, 4449524, 45908102, 12991445, 8428935, 13831558, 24906548, 18174210, 75636923, 25938221] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [15692098, 834336, 19487477, 31324978, 1840993, 27889071, 11457219, 30786800, 34733465, 25643818] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [15692098, 23908217, 4449524, 45908102, 834336, 12991445, 8428935, 19487477, 13831558, 24906548] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid + rerank@20: [18750453, 23908217, 18174210, 25938221, 9254550, 15692098, 23670644, 75636923, 25420421, 12991445] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['18750453']

Observation:
  The reranker improves nDCG@10: Hybrid 0.0000 vs reranked 1.0000.
  Relevant document 18750453 moves from hybrid-top-20 rank 14 to final rank 1.

Hypothesis:
  The cross-encoder's pairwise scores disagree with the hybrid fusion ranking in this case, which may explain the reordering; hypothesis only.

### Query 1180: The PRR MDA5 is a sensor of RNA virus infection.

Relevant document(s):
- 31272411 (grade 1): Immune signaling by RIG-I-like receptors. The RIG-I-like receptors (RLRs) RIG-I, MDA5, and LGP2 play a major role in pathogen sensing of RNA virus infection to initiate and modulate antiviral immunity... — exact query-token overlap 7 (ratio 0.70)

Per-method top-10:
- BM25: [10627801, 2566674, 16058322, 40044800, 14474178, 44366096, 26133404, 4402497, 31272411, 6501747] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.1111, ndcg@10=0.3010 — first relevant rank 9 — relevant in top-10: ['31272411']
- Dense: [24221369, 20725212, 8005007, 44366096, 9161988, 1383826, 9317504, 1970884, 8512633, 2566674] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [2566674, 44366096, 24221369, 10627801, 8005007, 20725212, 1970884, 9161988, 31272411, 1383826] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.1111, ndcg@10=0.3010 — first relevant rank 9 — relevant in top-10: ['31272411']
- Hybrid + rerank@20: [31272411, 44366096, 10627801, 6501747, 2566674, 15419873, 9591368, 16058322, 24221369, 8005007] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['31272411']

Observation:
  The reranker improves nDCG@10: Hybrid 0.3010 vs reranked 1.0000.
  Relevant document 31272411 moves from hybrid-top-20 rank 9 to final rank 1.

Hypothesis:
  The cross-encoder's pairwise scores disagree with the hybrid fusion ranking in this case, which may explain the reordering; hypothesis only.

## Reranker Hurts

Qualifying queries: 49; shown: 3.

### Query 70: Activation of PPM1D suppresses p53 function.

Relevant document(s):
- 4414547 (grade 1): Mosaic PPM1D mutations are associated with predisposition to breast and ovarian cancer Improved sequencing technologies offer unprecedented opportunities for investigating the role of rare genetic var... — exact query-token overlap 3 (ratio 0.50)
- 5956380 (grade 1): Exome sequencing identifies somatic gain-of-function PPM1D mutations in brainstem gliomas Gliomas arising in the brainstem and thalamus are devastating tumors that are difficult to surgically resect.... — exact query-token overlap 4 (ratio 0.67)

Per-method top-10:
- BM25: [5956380, 4414547, 24721866, 23985464, 7225911, 4389394, 4460880, 43880096, 25682129, 11903247] — precision@10=0.2000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['5956380', '4414547']
- Dense: [9483851, 27949347, 11903247, 6896063, 28107602, 31624828, 9169645, 21557055, 37204802, 14460402] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [5956380, 9483851, 11903247, 4414547, 6896063, 9169645, 27949347, 4389394, 43880096, 21307488] — precision@10=0.2000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=0.8772 — first relevant rank 1 — relevant in top-10: ['5956380', '4414547']
- Hybrid + rerank@20: [21307488, 6896063, 21557055, 31624828, 21521236, 11903247, 9483851, 4389394, 24721866, 5254463] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []

Observation:
  The reranker hurts nDCG@10: Hybrid 0.8772 vs reranked 0.0000.
  Relevant document 4414547 moves from hybrid-top-20 rank 4 to final rank None.
  Relevant document 5956380 moves from hybrid-top-20 rank 1 to final rank None.

Hypothesis:
  The cross-encoder's pairwise scores may promote a spuriously scored candidate above the relevant one; hypothesis only.

### Query 94: Albendazole is used to treat lymphatic filariasis.

Relevant document(s):
- 1215116 (grade 1): “Rapid-Impact Interventions”: How a Policy of Integrated Control for Africa's Neglected Tropical Diseases Could Benefit the Poor Over the past two decades there have been significant achievements in t... — exact query-token overlap 3 (ratio 0.43)

Per-method top-10:
- BM25: [17327939, 46926352, 3896759, 13070316, 46765242, 22613657, 11481946, 27240699, 27567994, 29634262] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [18321590, 20428155, 1215116, 6503185, 39424916, 11441172, 10617916, 84085333, 3935126, 10703001] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.3333, ndcg@10=0.5000 — first relevant rank 3 — relevant in top-10: ['1215116']
- Hybrid Weighted: [1215116, 17327939, 18321590, 46926352, 20428155, 3896759, 13070316, 6503185, 39424916, 11441172] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['1215116']
- Hybrid + rerank@20: [39424916, 25388309, 3935126, 20428155, 17327939, 46765242, 10703001, 11441172, 13070316, 1215116] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.1000, ndcg@10=0.2891 — first relevant rank 10 — relevant in top-10: ['1215116']

Observation:
  The reranker hurts nDCG@10: Hybrid 1.0000 vs reranked 0.2891.
  Relevant document 1215116 moves from hybrid-top-20 rank 1 to final rank 10.

Hypothesis:
  The cross-encoder's pairwise scores may promote a spuriously scored candidate above the relevant one; hypothesis only.

### Query 1278: The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events.

Relevant document(s):
- 11335781 (grade 1): Is autoimmunity the Achilles' heel of cancer immunotherapy? The emergence of immuno-oncology as the first broadly successful strategy for metastatic cancer will require clinicians to integrate this ne... — exact query-token overlap 5 (ratio 0.33)

Per-method top-10:
- BM25: [3825750, 40632104, 5691302, 11939159, 52180874, 5698494, 13256155, 14361849, 5641851, 46266579] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [11335781, 22635278, 4468861, 12705056, 22968257, 9767444, 10162553, 7343711, 24069089, 1196631] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['11335781']
- Hybrid Weighted: [11335781, 3825750, 40632104, 4468861, 5691302, 11939159, 52180874, 5698494, 13256155, 14361849] — precision@10=0.1000, recall@10=1.0000, mrr@10=1.0000, ndcg@10=1.0000 — first relevant rank 1 — relevant in top-10: ['11335781']
- Hybrid + rerank@20: [1454773, 4468861, 11254040, 52180874, 3825750, 22635278, 11335781, 11939159, 40632104, 341324] — precision@10=0.1000, recall@10=1.0000, mrr@10=0.1429, ndcg@10=0.3333 — first relevant rank 7 — relevant in top-10: ['11335781']

Observation:
  The reranker hurts nDCG@10: Hybrid 1.0000 vs reranked 0.3333.
  Relevant document 11335781 moves from hybrid-top-20 rank 1 to final rank 7.

Hypothesis:
  The cross-encoder's pairwise scores may promote a spuriously scored candidate above the relevant one; hypothesis only.

## All Methods Fail

Qualifying queries: 37; shown: 3.

### Query 1049: Ribosomopathies have a low degree of cell and tissue specific pathology.

Relevant document(s):
- 12486491 (grade 1): Ribosome-Mediated Specificity in Hox mRNA Translation and Vertebrate Tissue Patterning Historically, the ribosome has been viewed as a complex ribozyme with constitutive rather than regulatory capacit... — exact query-token overlap 4 (ratio 0.36)

Per-method top-10:
- BM25: [8925851, 24737389, 10669582, 21884449, 2888272, 11935250, 26112624, 12232678, 20418809, 38355793] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [24737389, 6670101, 26112624, 8925851, 8126244, 13915464, 8327914, 20018321, 13552682, 23342686] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [8925851, 24737389, 26112624, 6670101, 8126244, 13915464, 8327914, 20018321, 10669582, 21884449] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid + rerank@20: [8925851, 7487927, 24737389, 6670101, 10669582, 26112624, 22896970, 26495128, 13552682, 8426046] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []

Observation:
  Recall@10 is 0 for BM25, Dense, Hybrid Weighted, and the reranked pipeline.

Hypothesis:
  None of the four methods retrieves a relevant document in the top 10 for this query. No relevant document reaches the reranker candidate pool.

### Query 1110: Suboptimal nutrition is not predictive of chronic disease

Relevant document(s):
- 13770184 (grade 1): Global, regional, and national comparative risk assessment of 79 behavioural, environmental and occupational, and metabolic risks or clusters of risks, 1990–2015: a systematic analysis for the Global... — exact query-token overlap 3 (ratio 0.38)

Per-method top-10:
- BM25: [10010651, 31562330, 49429882, 22236223, 2032877, 43128141, 23983289, 18988265, 20188586, 756887] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [20288322, 4345757, 36558211, 70704988, 24273592, 9244474, 41310252, 8529693, 71341302, 27449472] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [10010651, 20288322, 49429882, 31562330, 4345757, 22236223, 2032877, 75636923, 36558211, 43128141] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid + rerank@20: [18988265, 9244474, 22236223, 2032877, 20188586, 756887, 41310252, 12122482, 16737163, 75636923] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []

Observation:
  Recall@10 is 0 for BM25, Dense, Hybrid Weighted, and the reranked pipeline.

Hypothesis:
  None of the four methods retrieves a relevant document in the top 10 for this query. No relevant document reaches the reranker candidate pool.

### Query 1175: The PPR MDA5 has two N-terminal CARD domains.

Relevant document(s):
- 31272411 (grade 1): Immune signaling by RIG-I-like receptors. The RIG-I-like receptors (RLRs) RIG-I, MDA5, and LGP2 play a major role in pathogen sensing of RNA virus infection to initiate and modulate antiviral immunity... — exact query-token overlap 1 (ratio 0.12)

Per-method top-10:
- BM25: [16058322, 38899659, 22191759, 22495397, 10627801, 36540079, 17402386, 8646760, 7751726, 207972] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Dense: [20725212, 18467982, 31107919, 3610282, 3127341, 11256632, 16058322, 16270577, 4321947, 31387717] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid Weighted: [16058322, 20725212, 38899659, 22191759, 18467982, 31107919, 3610282, 22495397, 3127341, 17402386] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []
- Hybrid + rerank@20: [22191759, 17402386, 16058322, 10627801, 31107919, 9169645, 612002, 31387717, 38899659, 22495397] — precision@10=0.0000, recall@10=0.0000, mrr@10=0.0000, ndcg@10=0.0000 — first relevant rank None — relevant in top-10: []

Observation:
  Recall@10 is 0 for BM25, Dense, Hybrid Weighted, and the reranked pipeline.

Hypothesis:
  None of the four methods retrieves a relevant document in the top 10 for this query. No relevant document reaches the reranker candidate pool.
