# RAG System Evaluation Report

**Date**: 2026-02-19 21:26
**Total Queries Evaluated**: 3

## Query Set Design

This evaluation uses a fixed query set stored at `src/eval/query_set.json`.

- Query type counts: {'direct': 11, 'edge_case': 5, 'synthesis': 6}
- Difficulty counts: {'easy': 2, 'hard': 9, 'medium': 11}

The set includes direct questions, cross-source synthesis questions, and edge cases that should trigger explicit missing-evidence behavior.

## System Configuration (This Run)

- Generator model: gemini-2.5-flash
- Retrieval configuration snapshot: {'k': 10, 'k_raw': 60, 'use_hybrid': False, 'use_reranking': False, 'top_k_after_rerank': 5, 'vector_weight': 0.5, 'bm25_weight': 0.5, 'year_min': None, 'year_max': None, 'source_types': None}

## Metrics

Groundedness is judged against retrieved evidence. Citation precision checks whether citations in the answer match retrieved chunks. Answer relevance rates whether the answer addresses the question.

## Implementation Notes

Retrieval uses a hybrid of semantic vector search and lexical BM25. Answers are generated with strict citation constraints that only allow citing retrieved chunks, and a repair pass runs when the model outputs citations outside the allowed set.

## Summary Metrics (This Run)

| Metric | Average Score | Notes |
|--------|---------------|-------|
| Groundedness | 2.33/4 | LLM-judged faithfulness to retrieved evidence |
| Citation Precision | 100.00% | Fraction of citations that match retrieved chunks |
| Answer Relevance | 3.67/4 | LLM-judged relevance to the question |

## Breakdown by Query Type

| query_type   |   groundedness_avg |   citation_precision_avg |   relevance_avg |   count |
|:-------------|-------------------:|-------------------------:|----------------:|--------:|
| direct       |            2.33333 |                        1 |         3.66667 |       3 |

## Per-Query Summary

| query_id   | query_type   |   groundedness_score | citation_precision_value   |   answer_relevance_score |
|:-----------|:-------------|---------------------:|:---------------------------|-------------------------:|
| Q01        | direct       |                    4 | 100.00%                    |                        3 |
| Q02        | direct       |                    2 | 100.00%                    |                        4 |
| Q03        | direct       |                    1 | 100.00%                    |                        4 |

## Best Performing Queries


**Q01**: What architectural components are critical for operational city digital twins?
- Groundedness: 4 | Citation precision: 100.00% | Relevance: 3

**Q02**: What benchmark datasets exist for evaluating city digital twin systems?
- Groundedness: 2 | Citation precision: 100.00% | Relevance: 4

**Q03**: How is real-time sensor data integrated into digital twin platforms?
- Groundedness: 1 | Citation precision: 100.00% | Relevance: 4


## Failure Cases (Representative)


**Q03**: How is real-time sensor data integrated into digital twin platforms?
- Groundedness: 1 | Citation precision: 100.00% | Relevance: 4
- Grounding note: The answer includes multiple fabricated citations (Abdelrahman 2025, chunk_96 and Alkhateeb 2023, chunk_07) which are not present in the provided evidence. Some claims associated with these fabricated citations are therefore ungrounded or cannot be verified against the given evidence.
- Invalid citations: []
- Evidence snippets:
  - (Alkhateeb2023, chunk_45): y making them a digital twin of each other. These two datasets combined can enable the development and evaluation of digital twin-aided applications in real-world communication systems. Communication-Sensing Trade-Off: In order to generate an accurate real-time digital twin with minimal latency, the sensing data collected across different devices, in some cases, need to be transferred quickly to a central unit for further processing (digital twin levels 2 and 3). The data transfer rate is dependent on the available bandwidth of the communication system itself. As the amount of sensing data increases (for example, with the increase in sensing modalities or the number of devices), so does the requirement for communication bandwidth. While access to diverse and detailed sensing information ca
  - (Barbie2023, chunk_51): l Twin Prototype, the team also gets a digital twin that can be utilized to monitor the physical twin during operation and collect data without the need to physically connect to the sensor bar mounted on a tractor. This enables the development of embedded software systems without the need to physically connect to the hardware and hence reduces costs that may be needed for spare hardware otherwise. Without the need for hardware in the development loop, the seasonal data gathering missions become less of a problem. Only the last point cannot be solved utilizing a digital twin, as the software modules are external features. The problem at that point is the embedded software community that develops device drivers with tight coupling to the overall system, e. g. a specific middleware. However,

**Q02**: What benchmark datasets exist for evaluating city digital twin systems?
- Groundedness: 2 | Citation precision: 100.00% | Relevance: 4
- Grounding note: The first sentence correctly states that the evidence does not describe any specific benchmark datasets, which is supported by a review of the chunks. However, the second sentence contains a hallucinated claim that "Assessing and benchmarking 3D city models" is mentioned as an area of research, as this phrase does not appear in the provided evidence. Furthermore, the citation to (Luo 2024, chunk_128) is a fabricated citation, as this chunk is not provided in the evidence, and the citation to (Lei 2023, chunk_95) for that phrase is incorrect.
- Invalid citations: []
- Evidence snippets:
  - (White2021, chunk_07): mation on the object/ system. As the object/system increases in complexity a digital twin may be identical in only relevant areas and have only the real-time data necessary to support any desired simulations. How accurate and useful a digital twin is, depends on the level of detail put into it and how comprehensive the available data is. Digital twins allow for the simulation of many options before taking physical action in the real world to identify the strengths and weaknesses of each plan. This is especially important in safety critical * Corresponding author. E-mail addresses: whiteg5@scss. tcd. ie (G. White), zinka@tcd. ie (A. Zink), lara. codeca@tcd. ie (L. Codec´ a), siobhan. clarke@scss. tcd. ie (S. Clarke). Contents lists available at Science Direct Cities journal homepage: www. e
  - (Lei2023, chunk_95): hank Dr Claire Ellul (University College London)forhersupportanddiscussionsrelatedtothesurvey. Wethank the members of the NUS Urban Analytics Lab for the discussions, in particular, Dr Pengyuan Liu for the constructive comments on the early versions of the paper. This research is part of the projects (i) Largescale3DGeospatial Datafor Urban Analytics, whichissupportedbythe National University of Singapore under the Start Up Grant R-295-000171-133; and (ii) Multi-scale Digital Twins for the Urban Environment: From Heartbeatsto Cities, whichissupportedbythe Singapore Ministry of Education Academic Research Fund Tier 1. References [1] Ehab Shahat, Chang T. Hyun, Chunho Yeom, City digital twin potentials: A review and research agenda, Sustainability 13 (6) (2021) 3386, http://dx. doi. org/10.3


## Run Logs

Machine-readable runs: outputs/eval/eval_runs_baseline_20260219_212525.jsonl


## Limitations and Next Steps

The evaluation uses an LLM judge, so scores can be noisy. For follow-up, consider adding deterministic retrieval metrics, expanding edge cases for missing evidence, and validating citation alignment against longer evidence spans.


## Notes

To quantify the impact of enhancements, run evaluation with `--compare` to produce baseline and enhanced runs and include the Enhancement Impact section.
