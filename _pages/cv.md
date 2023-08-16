---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

[Professional Resume (PDF)](/files/resume/resume_20230816.pdf){:target="_blank" .btn .btn--primary .btn--large}

# Education

* M.S. in Computer Science, University of Massachusetts Amherst, Amherst, MA, USA. 2024
* B.Tech in Electrical Engineering, Indian Institute of Technology Delhi, New Delhi, India. 2019 

# Research experience

## Graduate Student Researcher, UMass Amherst
Sep 2022 - Present

### Autonomous Learning Laboratory

  * Developed off-policy algorithms to learn personalized sepsis treatment policies from historical ICU data of over 17K patients. 
  * Contributed to the Seldonian ML Toolkit, an open-source Python library for safe and interpretable machine learning.
  * Used the Seldonian framework to learn policies that perform higher than current practice and provide safety guarantees.

### Natural Language Processing Group
  
  * Conducted a study to evaluate the ability of large vision-language models to perform comparative reasoning.
  * Curated a novel dataset to evaluate multi-modal reasoning capabilities of vision-enabled large language models like GPT-4.
  * Developed a self-supervised prompt engineering method to achieve 53.7% accuracy for zero-shot comparative reasoning.

## Undergraduate Student Researcher, IIT Delhi
May 2017 - Jun 2019
  
### Centre for Biomedical Engineering

Summer design research grant. Adapting the equipment on trains to measure vibration data and detect faults in railway tracks.

### Computer Technology Group

Bachelor's Thesis. Predicting occupancy in commercial spaces to automate the HVAC system using Hidden Markov Models.

### Centre for Applied Research in Electronics
    
Research grant by Voxomos Systems. Trained an LSTM recurrent neural network based chatbot for customer care services.

# Professional experience

## Machine Learning Engineer, Reliance Jio Infocomm Ltd.
Jul 2021 - Jul 2022

Led the initiative to develop cost-effective solutions for improving the 4G experience for customers and providing network insights.
    
* Coordinated a team of 6 to design a statistical model using response surface methodology for optimizing network coverage.
* Attained a 15% improvement in network throughput predictions by implementing a Mixture of Experts model with PyTorch.
* Overhauled the call drop RCA pipeline by utilizing SHAP scores resulting in 18% fewer false positives in downstream tasks.

## Data Scientist, Reliance Jio Infocomm Ltd.
Jul 2019 - Jun 2021

Used machine learning and Bayesian statistics to gain actionable insights into customer behavior and refine retention strategies. 

* Deployed a distributed model with Apache Spark and MS Azure to perform churn prediction for 400M+ daily customers.
* Curated over 1100 features for churn prediction to improve customer retention resulting in $35M/mo in additional revenue.
* Collaborated with vendors to launch A/B testing of ad campaigns on geographic cohorts and created a Tableau dashborad.
* Reduced data pre-processing time by 70% during ETL by implementing a multiprocessing pipeline for feature extraction.

## Intern, Claym Media and Tech Pvt. Ltd.
Summer 2018

Developed an automated solution for analyzing financial reports and detecting fraudulent transactions in mobile advertising.

* Created a low-latency pipeline in Java for the AI-driven framework to flag potentially fraudulent transactions in real time.
* Standardized reports of 50M+ monthly transactions from 20+ sources by regex string manipulation in a modular framework. 

# Publications

  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
# Teaching

  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
# Leadership and Awards

* Highest annual performance feedback rating (A*) for 3 years in a row, Reliance Jio, 2020-22
* Outstanding Technical Project, AI CoE, Reliance Jio, 2019
* Student Mentor, Mentorship Committee, IIT Delhi, 2018-19
* Secretary, Electrical Engineering Society, IIT Delhi, 2017-18
* Director, Theater Society, IIT Delhi, 2017-18
* JEE Advanced - National Rank 321 among 1.5 million total applicants, 2015
* KVPY Fellow - Awarded scholarship by Indian Institute of Science under the KVPY program, 2014
