---
title: "Microsoft ML Wrappers"
excerpt: "A unified wrapper for various ML frameworks - to have one uniform scikit-learn format for predict and predict_proba functions"
collection: portfolio
date: 2024-02-28 20:30:00 -0500
last_modified_at: 2024-06-06 08:30:00 -0500
---

Responsible AI tools should be able to work with a broad spectrum of machine learning models and datasets. Much of this
functionality is based on the ability to call `predict` or `predict_proba` on a model and get back the predicted values
or probabilities in a specific format.

However, there are many different models outside of scikit-learn and even within scikit-learn which have unusual
outputs or require the input in a specific format. Some, like pytorch, don't even have the `predict`/`predict_proba`
function specification.

These wrappers handle a variety of frameworks, including pytorch, tensorflow, keras wrappers on tensorflow, variations
on scikit-learn models (such as the SVC classification model that doesn't have a `predict_proba` function), lightgbm
and xgboost, as well as certain strange pipelines we have encountered from customers and internal users in the past.

## Contributions

- **Extended LLM Support**: Added the ability to specify the system prompt and chat history for LLMs.
- **Asynchronous OpenAI Execution**: Added support for asynchronous execution of OpenAI models, with a rate limiter to
  prevent going over the API limits.
- **Compatibility Improvements**: Expanded the compatibility of the wrapper to include all versions of the OpenAI API,
from legacy to the latest.

[Website](https://ml-wrappers.readthedocs.io/en/latest/index.html) \| [GitHub](https://github.com/microsoft/ml-wrappers)