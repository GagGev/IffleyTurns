# Rare disease similarity detection
Finding common patterns between rare diseases.

## Overview
The project aims to detect and outline relationships between various rare diseases. The goal of the project is to provide a comprehensive, scientifically justifiable rare disease relationship graph, allowing researchers to explore potential drug discovery pathways using the relationships.

### Definitions
Rare disease - A disease affecting less than 1 in 2,000 people

## Data
### Where to look for information
- Rare-disease descriptions and classifications:
  - NORD Rare Disease Database: https://rarediseases.org/rare-diseases/
  - Orphanet: https://www.orpha.net/
  - NIH Genetic and Rare Diseases Information Center: https://rarediseases.info.nih.gov/
- Phenotypes and disease terminology:
  - Human Phenotype Ontology (HPO): https://hpo.jax.org/
  - Mondo Disease Ontology: https://mondo.monarchinitiative.org/
- Genes and genetic variants:
  - OMIM: https://www.omim.org/
  - ClinGen: https://clinicalgenome.org/
  - ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- Biological pathways and drug targets:
  - Reactome: https://reactome.org/
  - Open Targets Platform: https://platform.opentargets.org/
  - ChEMBL: https://www.ebi.ac.uk/chembl/
- Clinical and regulatory evidence:
  - ClinicalTrials.gov: https://clinicaltrials.gov/
  - FDA Orphan Drug Designations: https://www.accessdata.fda.gov/scripts/opdlisting/oopd/
  - EMA Orphan Medicines: https://www.ema.europa.eu/en/human-regulatory-overview/orphan-designation-overview
- Scientific literature:
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/
  - Europe PMC: https://europepmc.org/

### What ML features to generate
- Disease classification:
  - Shared disease categories and ontology parents
  - Affected organs and body systems
- Clinical characteristics:
  - Number and proportion of shared HPO phenotypes
  - Age-of-onset similarity
  - Shared major symptoms and complications
- Genetics:
  - Number and proportion of associated genes in common
  - Shared inheritance pattern
- Epidemiology:
  - Prevalence similarity
  - Similarity in affected sexes, populations, and regions
- Treatments:
  - Shared approved drugs and treatment types
- The extent to which the disease was 

#### Potential expansions
- Semantic phenotype similarity using the HPO hierarchy and term rarity
- Shared biological pathways and molecular mechanisms
- Common drug targets and protein-interaction network proximity
- Similarity in disease progression and symptom order over time
- Gene-expression, proteomic, and metabolomic profile similarity
- Similarity between clinical trials, including interventions and outcomes
- Comorbidity and diagnostic-pathway similarity from de-identified health records
- Text embeddings generated from disease descriptions and clinical documents

## Methods
Approach 1: Calculate a 'distance' between two given diseases
Approach 2: Given a disease, determine distance from all other diseases
Approach 3: Map the diseases into a multidimensional space, similar to embeddings

### Simple approach
Use the generated features to create a simple "distance" metric that determines how close the two diseases are by assigning an "importance" score to each of the feature types.

### ML approach
Coming soon

## Evaluation
There is no single correct measure of disease similarity. The initial evaluation should therefore focus on whether the method produces sensible, explainable results supported by trusted sources.

### Initial evaluation
1. Select a small and varied set of rare diseases.
2. Create a short list of disease pairs that are already known to be related and cite the evidence for each relationship.
3. Include several clearly unrelated disease pairs as controls.
4. Calculate the similarity score for every pair.
5. Check whether known related diseases generally receive higher scores than the controls.
6. For each result, display the features that contributed to the score and link them to their data sources.
7. Manually review the five closest matches for each example disease and record whether each result is supported, plausible, or unsupported.

The initial approach will be considered successful if it consistently ranks known relationships above unrelated controls and provides a clear, source-backed explanation for every score.

### Potential expansions
- Evaluate the approach on a larger benchmark dataset.
- Introduce ranking metrics such as Precision@10 and nDCG.
- Compare the results against alternative similarity methods.
- Ask domain experts to review novel predicted relationships.
- Use historical data to test whether the method could have anticipated relationships discovered later.
