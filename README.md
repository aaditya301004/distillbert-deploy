## Overview
This project focuses on detecting fake job postings using a fine-tuned **DistilBERT** model, optimized for natural language understanding. It leverages a **semi-supervised learning pipeline** that combines labeled datasets from Kaggle and OpenML with over **200,000 unlabeled real-world job listings** scraped from platforms like Naukri.com. The goal is to build a robust, scalable, and real-world-applicable model that flags suspicious job postings based on content semantics, structure, and red flag indicators.


## Features

- **DistilBERT-Based Classification**  
  Utilizes the lightweight transformer model DistilBERT for high-speed, context-aware job posting classification.

- **Semi-Supervised Learning with Pseudo-Labeling**  
  Unlabeled real job data (~2 lakh entries) are automatically assigned labels using the model’s confidence thresholds to improve generalization.

- **Custom Evaluation Metrics**  
  - Model Conifdence
  - Title-description mismatch detection
  - Skill and description consistency

- **Streamlit Frontend**  
  Real-time web interface for fraud prediction and score visualization using the trained DistilBERT model.

- **Scalable Data Pipeline**  
  Modular structure supports continuous ingestion of new scraped job listings for retraining and evaluation.


## Data Preparation

This project combines labeled datasets from public sources with a large volume of real-world, unlabeled job postings collected from Indian job portals to create a robust training set.

### 1. Labeled Data

- **Sources**:
  - Kaggle Fake Job Postings Dataset  
  - OpenML Job Fraud Detection Dataset

- **Size**: ~55,000 labeled records  
- **Structure**:
  - `job_title`, `job_description`, `skills`, `company_profile`, `employment_type`, `location`, `salary`, `label`  
  - Labels: `0` for real, `1` for fake

- **Cleaning Tasks**:
  - Removed duplicates and null entries  
  - Normalized text fields (lowercase, punctuation removal)  
  - Standardized column names and value formats


### 2. Unlabeled Real-World Data

- **Source**: Scraped from **Naukri.com** and other job platforms.
- **Volume**: ~1,50,000 real, unlabeled job postings

- **Collected Fields**:
  - `job_title`, `job_description`, `skills_required`, `employment_type`, `location`, `salary_range`, `industry`, `company_profile`

- **Cleaning & Alignment**:
  - Removed boilerplate/template-like descriptions  
  - Filtered out entries with missing critical fields  
  - Converted structure to match the labeled data format for pseudo-labeling compatibility  
  - Ensured linguistic diversity across roles, domains, and locations

This hybrid dataset enables the model to learn both from clearly labeled examples and unlabeled real-world patterns using pseudo-labeling.

