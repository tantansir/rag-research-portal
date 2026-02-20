# RAG System Evaluation Report

**Date**: 2026-02-19 23:14
**Total Queries Evaluated**: 22

## Query Set Design

This evaluation uses a fixed query set stored at `src/eval/query_set.json`.

- Query type counts: {'direct': 11, 'edge_case': 5, 'synthesis': 6}
- Difficulty counts: {'easy': 2, 'hard': 9, 'medium': 11}

The set includes direct questions, cross-source synthesis questions, and edge cases that should trigger explicit missing-evidence behavior.

## System Configuration (This Run)

- Generator model: gemini-2.5-flash
- Retrieval configuration snapshot: {'k': 10, 'k_raw': 60, 'use_hybrid': True, 'use_reranking': True, 'top_k_after_rerank': 5, 'vector_weight': 0.5, 'bm25_weight': 0.5, 'year_min': None, 'year_max': None, 'source_types': None}

## Metrics

Groundedness is judged against retrieved evidence. Citation precision checks whether citations in the answer match retrieved chunks. Answer relevance rates whether the answer addresses the question.

## Implementation Notes

Retrieval uses a hybrid of semantic vector search and lexical BM25. Answers are generated with strict citation constraints that only allow citing retrieved chunks, and a repair pass runs when the model outputs citations outside the allowed set.

## Summary Metrics (This Run)

| Metric | Average Score | Notes |
|--------|---------------|-------|
| Groundedness | 3.91/4 | LLM-judged faithfulness to retrieved evidence |
| Citation Precision | 99.65% | Fraction of citations that match retrieved chunks |
| Answer Relevance | 3.55/4 | LLM-judged relevance to the question |

## Enhancement Impact (Baseline vs Enhanced)

Baseline settings: vector-only retrieval, no LLM reranking.

Enhanced settings: hybrid retrieval (BM25 + vector) plus LLM reranking.

| Metric | Baseline | Enhanced | Delta |
|--------|----------:|---------:|------:|
| Groundedness (avg /4) | 2.91 | 3.91 | +1.00 |
| Citation Precision (avg) | 99.43% | 99.65% | +0.22% |
| Answer Relevance (avg /4) | 3.41 | 3.55 | +0.14 |

### Enhancement Impact by Query Type

        | query_type   |   groundedness_baseline | citation_precision_baseline   |   relevance_baseline |   groundedness_enhanced | citation_precision_enhanced   |   relevance_enhanced |   groundedness_delta | citation_precision_delta   |   relevance_delta |
|:-------------|------------------------:|:------------------------------|---------------------:|------------------------:|:------------------------------|---------------------:|---------------------:|:---------------------------|------------------:|
| direct       |                    2.91 | 100.00%                       |                 3.27 |                    3.91 | 100.00%                       |                 3.36 |                 1    | 0.00%                      |              0.09 |
| edge_case    |                    2.6  | 100.00%                       |                 3.8  |                    3.8  | 100.00%                       |                 3.4  |                 1.2  | 0.00%                      |             -0.4  |
| synthesis    |                    3.17 | 97.92%                        |                 3.33 |                    4    | 98.72%                        |                 4    |                 0.83 | 0.80%                      |              0.67 |

        ## Breakdown by Query Type

| query_type   |   groundedness_avg |   citation_precision_avg |   relevance_avg |   count |
|:-------------|-------------------:|-------------------------:|----------------:|--------:|
| direct       |            3.90909 |                 1        |         3.36364 |      11 |
| edge_case    |            3.8     |                 1        |         3.4     |       5 |
| synthesis    |            4       |                 0.987179 |         4       |       6 |

## Per-Query Summary

| query_id   | query_type   |   groundedness_score | citation_precision_value   |   answer_relevance_score |
|:-----------|:-------------|---------------------:|:---------------------------|-------------------------:|
| Q01        | direct       |                    4 | 100.00%                    |                        4 |
| Q02        | direct       |                    4 | 100.00%                    |                        4 |
| Q03        | direct       |                    4 | 100.00%                    |                        4 |
| Q04        | direct       |                    4 | 100.00%                    |                        2 |
| Q05        | direct       |                    4 | 100.00%                    |                        4 |
| Q06        | direct       |                    4 | 100.00%                    |                        4 |
| Q07        | direct       |                    4 | 100.00%                    |                        3 |
| Q08        | direct       |                    4 | 100.00%                    |                        4 |
| Q09        | direct       |                    3 | 100.00%                    |                        2 |
| Q10        | direct       |                    4 | 100.00%                    |                        4 |
| Q11        | synthesis    |                    4 | 92.31%                     |                        4 |
| Q12        | synthesis    |                    4 | 100.00%                    |                        4 |
| Q13        | synthesis    |                    4 | 100.00%                    |                        4 |
| Q14        | synthesis    |                    4 | 100.00%                    |                        4 |
| Q15        | synthesis    |                    4 | 100.00%                    |                        4 |
| Q16        | edge_case    |                    4 | 100.00%                    |                        4 |
| Q17        | edge_case    |                    4 | 100.00%                    |                        4 |
| Q18        | edge_case    |                    3 | 100.00%                    |                        4 |
| Q19        | edge_case    |                    4 | 100.00%                    |                        3 |
| Q20        | edge_case    |                    4 | 100.00%                    |                        2 |
| Q21        | direct       |                    4 | 100.00%                    |                        2 |
| Q22        | synthesis    |                    4 | 100.00%                    |                        4 |

## Best Performing Queries


**Q01**: What architectural components are critical for operational city digital twins?
- Groundedness: 4 | Citation precision: 100.00% | Relevance: 4

**Q02**: What benchmark datasets exist for evaluating city digital twin systems?
- Groundedness: 4 | Citation precision: 100.00% | Relevance: 4

**Q03**: How is real-time sensor data integrated into digital twin platforms?
- Groundedness: 4 | Citation precision: 100.00% | Relevance: 4


## Failure Cases (Representative)


**Q09**: What data pipeline architectures are commonly used in digital twin systems?
- Groundedness: 3 | Citation precision: 100.00% | Relevance: 2
- Grounding note: Most claims are directly supported by the evidence with correct citations. However, one minor gap exists: the claim "The integration of sensor data, data analytics, and machine learning allows for..." cites (White 2021, chunk_06), but this chunk explicitly mentions only "data analytics and machine learning" in the context of updating a digital twin, not "sensor data" directly within that specific sentence. While sensor data is critical for digital twins and mentioned in other evidence chunks, its inclusion and citation against White 2021, chunk_06 for this specific claim is not fully direct.
- Invalid citations: []
- Evidence snippets:
  - (Abdelrahman2025, chunk_71): nd practice due to technical, organizational, and data-related aspects [30,133]. For instance, the integration of BIM and real-time data streams from Io T sensors is complex [134], interoperability between different data schemas, sources, software and platforms is difficult, in addition to scalability issues [130,135,136]. Architectural domain Architecture terms show significance values closer to the building terms. Digital Twins are used in Architecture for various purposes, such as design and planning, construction management, and operation and maintenance. The architecture domain is characterized by the prominent use of 2D/3D data (Residual of 2.8), data representation (residual of 2), and validation (residual of 1.3) (Fig. 11 - b). The negative residual of simulation models indicates t
  - (Abdelrahman2025, chunk_13): struction (AEC) domain include (4) Real-time capabilities, (5) Real-time update of the digital system with the physical system, and (6) Decision support [42–44] (Fig. 1). While these interpretations may not always provide a comprehensive understanding, and some may even collide, they can capture various aspects of digital twins relevant to the specific study where the definition is employed. By consolidating diverse perspectives on digital twins from a large number of sources, our goal is to formulate a datadriven, comprehensive, and standardized definition of digital twins that captures a unified understanding across the built environment, and bring us one step closer to a consensus. Considering the inconsistent terminology and contrasting implementations, recent studies have introduced t

**Q18**: Are there standardized APIs for interoperability between different digital twin platforms?
- Groundedness: 3 | Citation precision: 100.00% | Relevance: 4
- Grounding note: Most claims in the answer are directly supported by the evidence and correctly cited. However, one statement regarding the NGSI-LD API's function is slightly imprecise; the evidence attributes "modeling an Urban Digital Twin" to the NGSI-LD information model, while the answer attributes it directly to the API, though the API does enable interaction with this model.
- Invalid citations: []
- Evidence snippets:
  - (Mazzetto2024, chunk_128): ng Electronics Industries) [156] USA Digital twin product, manufacturing, and lifecycle frameworks Released IPC-2551, the first international standard for DTs, enabling interoperability across digital and physical entities. Stand ICT. eu 2026 [157] Europe ICT standardization in digital twin technologies Funded by the EU, aiming to streamline digital twin standardization efforts by providing a comprehensive landscape of global work in this area. Local Digital Twin & Citiverse EDIC [158] Europe (Estonia, Germany, Slovenia, Czech Republic, Spain) UDT and virtual worlds A European Commission initiative to support the deployment of local DTs and develop the Citiverse, emphasizing standardized, interoperable instruments. 5. Discussion In the discussion, the insights from both bibliometric and co
  - (WEF2022, chunk_92): nario-based targets for the standardization of digital twin technologies, taking into account their own business characteristics and digital twin applications. Eighteen research institutes, including Beijing University of Aeronautics and Astronautics and Shandong University, are already conducting research on digital twin standard systems. Source: Open data, collated by CAICT In terms of evaluation, there has not yet been a unified acceptance standard and operation evaluation system. The evaluation standards for acceptance, testing, trial runs and operations of each project are not uniform, and users face certain risks and difficulties in judging project completion and construction effects. There is still a need to create an evaluation system for the level of completion, and relevant asses

**Q04**: What human-centered design practices are used in digital twin interfaces?
- Groundedness: 4 | Citation precision: 100.00% | Relevance: 2
- Grounding note: Every factual claim in the answer is directly supported by the provided evidence and correctly cited. The answer accurately extracts information about public engagement, consensus, and the current lack of user-centered design in digital twin construction, reflecting the content of the Lei 2023 chunks. The concluding meta-commentary correctly assesses that the evidence discusses challenges and desired aspects rather than detailing specific human-centered design practices currently *used*, which is an accurate observation given the provided texts.
- Invalid citations: []
- Evidence snippets:
  - (Lei2023, chunk_56): l access around this challenge [20]. The construction of digital twins is typically driven by the interests of the developers, which is not a usercentred design [37]. As a result, not all types of users are considered in this process, such as vulnerable groups and nonexperts. Furthermore, it also questions to what extent participatory feedback would be reflected in digital twins and how the communication loop could be shaped through the interaction between human and digital twins [1,83]. Financing is crucial to operating digital twins. The first aspect is associated with equipment cost. In many cases, it refers to the cost of getting software licences, requiring commercial data, or purchasing devices [100]. For example, in the stage of collecting data, additional finances are needed to ins
  - (Lei2023, chunk_81): anges happening in the real world’ and ‘revise workflow to update digital twin models regularly’. From the social perspective, public engagement is highlighted, e. g. reflecting feedback on the digital models. Furthermore, it is significant to reach a consensus by determining the interests of various groups — what new understanding should be generated before beginning another cycle of digital twins. 5 Discussion Our results, combining the results of the systematic review and the Delphi survey, reveal a lengthy list of challenges pertaining to the design and implementation of urban digital twins. One of the key results is that the challenges identified in the literature and survey are mostly consistent. Reviewing recent papers exposes that technical challenges are discussed in more detail.


## Run Logs

Machine-readable runs: outputs/eval/eval_runs_enhanced_20260219_230030.jsonl


## Limitations and Next Steps

The evaluation uses an LLM judge, so scores can be noisy. For follow-up, consider adding deterministic retrieval metrics, expanding edge cases for missing evidence, and validating citation alignment against longer evidence spans.
