# Association between RD and incident depression

This repository contains the code used in our study:

**“Rheumatic Disease Multimorbidity and Graded Risk of Depression with Limited Inflammatory Mediation and Proteomic Stratification in the UK Biobank”**

---

## Project overview

This project investigates the association between rheumatic diseases (RDs) and incident depression using UK Biobank data. The analysis includes:

- Prospective cohort analysis (Cox models)
- Multimorbidity (RD count) analysis
- Inflammatory marker association and mediation
- Propensity score matching
- External validation (NHANES)
- Proteomics-based prediction using machine learning

Due to data access restrictions, raw data are not included. Scripts are provided for reproducibility given appropriate access to UK Biobank and NHANES data.

---

## Scripts description


### 1. Epidemiological analysis
- `Cox_anyRD.R`  
  Main Cox proportional hazards models for RD and depression risk.

- `Cox_nonliear.R`  
  Restricted cubic spline analysis for dose–response relationship.

---

### 2. Inflammation analysis
- `association_RD_inflammatory.R`  
  Test associations between inflammatory markers and RD / depression.

- `Cox_nonliear_mediator.R`  
  Restricted cubic spline analysis for dose–response relationship in inflammatory markers and incident depression.

- `mediation_analysis.R`  
  Bootstrap-based mediation analysis of inflammatory markers.

---

### 3. Matching analysis
- `matchit_RD.R`  
  Propensity score matching and re-analysis in matched cohort.

---

### 4. NHANES validation
- `logistic_NHANES.R`  
  External supporting analysis using NHANES data.

---

### 5. Proteomics and machine learning
- `GO enrichment.R`  
  Pathway enrichment analysis and plot.
  
- `machine learning`  
  All python scripts used for modeling and plotting.

---

## Notes

- UK Biobank data require approved access.
- NHANES data are publicly available.
- Some scripts depend on local data paths and may require adjustment.

---
 
