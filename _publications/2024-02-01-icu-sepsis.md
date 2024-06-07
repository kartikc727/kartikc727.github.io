---
title: "ICU-Sepsis: A Benchmark MDP Built from Real Medical Data"
collection: publications
permalink: /publication/2024-02-01-icu-sepsis
excerpt: 'We propose a novel benchmark MDP for sepsis treatment in the ICU built using medical data from real patients.'
date: 2024-02-01
venue: 'Reinforcement Learning Conference'
# paperurl: ''
# citation: ''
---

{% include figure image_path="images/assets/publications/icu-sepsis/sepsis-fig-timeline.png"
alt="Environment visualization" caption="Illustration of one episode in the ICU-Sepsis environment." %}

We present *ICU-Sepsis*, an environment that can be used in benchmarks for evaluating reinforcement learning (RL)
algorithms. Sepsis management is a complex task that has been an important topic in applied RL research in recent
years. Therefore, MDPs that model sepsis management can serve as part of a benchmark to evaluate RL algorithms on a
challenging real-world problem. However, creating usable MDPs that simulate sepsis care in the ICU remains a challenge
due to the complexities involved in acquiring and processing patient data. ICU-Sepsis is a lightweight environment that
models personalized care of sepsis patients in the ICU. The environment is a tabular MDP that is widely compatible and
is challenging even for state-of-the-art RL algorithms, making it a valuable tool for benchmarking their performance.
However, we emphasize that while ICU-Sepsis provides a standardized environment for evaluating RL algorithms, it should
not be used to draw conclusions that guide medical practice.

Accepted at the [Reinforcement Learning Conference](https://rl-conference.cc/), 2024.

Preprint: (coming soon)

Code: [GitHub](https://github.com/icu-sepsis/icu-sepsis)

Cite as:

{% raw %}
```
@inproceedings{
choudhary2024icusepsis,
title={{ICU}-Sepsis: A Benchmark {MDP} Built from Real Medical Data},
author={Kartik Choudhary and Dhawal Gupta and Philip S. Thomas},
booktitle={Reinforcement Learning Conference},
year={2024},
}
```
{% endraw %}
