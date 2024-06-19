---
title: "Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges"
collection: publications
permalink: /publication/2024-05-20-judging-judges
excerpt: 'A comprehensive study of the LLM-as-a-judge paradigm in a controlled setup that reveals new results about its strengths and weaknesses.'
date: 2024-05-20
venue: 'Under Review, NeurIPS'
---

{% include figure image_path="images/assets/publications/judging-judges/judge-score-alignment.png"
alt="Judge alignment chart" caption="(Left) Scores assigned by different judge LLMs to different exam-taker
LLMs. (Right) Alignment scores of different judges with human annotation." %}

Offering a promising solution to the scalability challenges associated with human evaluation, the LLM-as-a-judge
paradigm is rapidly gaining traction as an approach to evaluating large language models (LLMs). However, there are
still many open questions about the strengths and weaknesses of this paradigm, and what potential biases it may hold.
In this paper, we present a comprehensive study of the performance of various LLMs acting as judges. We leverage
TriviaQA as a benchmark for assessing objective knowledge reasoning of LLMs and evaluate them alongside human
annotations which we found to have a high inter-annotator agreement. Our study includes 9 judge models and 9 exam
taker models -- both base and instruction-tuned. We assess the judge model's alignment across different model sizes,
families, and judge prompts. Among other results, our research rediscovers the importance of using Cohen's kappa as a
metric of alignment as opposed to simple percent agreement, showing that judges with high percent agreement can still
assign vastly different scores. We find that both Llama-3 70B and GPT-4 Turbo have an excellent alignment with humans,
but in terms of ranking exam taker models, they are outperformed by both JudgeLM-7B and the lexical judge Contains,
which have up to 34 points lower human alignment. Through error analysis and various other studies, including the
effects of instruction length and leniency bias, we hope to provide valuable lessons for using LLMs as judges in
the future.

Under Review at [NeurIPS](https://neurips.cc/), 2024.

[Paper](https://arxiv.org/abs/2406.12624){:target="_blank" rel="noopener noreferrer"} \|
[Code](https://github.com/judging-judges/judging-judges){:target="_blank" rel="noopener noreferrer"}
