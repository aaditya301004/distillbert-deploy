## Overview
This project focuses on detecting fake job postings using a fine-tuned **DistilBERT** model, optimized for natural language understanding. It leverages a **semi-supervised learning pipeline** that combines labeled datasets from Kaggle and OpenML with over **200,000 unlabeled real-world job listings** scraped from platforms like Naukri.com. The goal is to build a robust, scalable, and real-world-applicable model that flags suspicious job postings based on content semantics, structure, and red flag indicators.

---

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

---


