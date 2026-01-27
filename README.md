# Simple-Contrastive-Learning-Embdedding-Model

## Introduction
Here we apply SimCLR architecture to a text based task using random word dropout to create randomised views of data instances to form an array of positive and negative samples for out contrastive task. The architecture follows closely with that using by (Chen, et al., 2020). This work acts as a foundational project towards understanding Contratsive Learning Techniques, in this case applied within the context of NLP. Evaluation of the work consists of a visual analysis of the embedding space using PCA.

## Method
The following architectural changes were made in order to accomodate text data:
  * A text encoder is used instead of a ResNet for f(*)
  * Semantic preserving text augmentations are used instead, in this case random word dropout with a probability p=0.2)
  * Mean pooling was used before passing embeddings to the projection head

Training is done for one epoch of the training_data subset of the hugging face wikitext-2 dataset. Due to the exploratory nature of the study further epochs were not required but form basis for further study.

## Evaluation
We evaluate the performance of the contrative learning model using PCA Component visualisation for embedding values for the first 500 data instance of the wikitext-2 training data subset. As the trained encoder is not used for a downstream task there was no requirement for quantitative analysis of the model. Plots for both the embedding and projection head outputs were done using the first two principle components for each respectively. These plots are shown in figures 1 and 2.


<img width="635" height="474" alt="Screenshot 2026-01-26 at 18 38 32" src="https://github.com/user-attachments/assets/777371ac-25f0-4af7-98a1-bd87c31eb4d2" />

*Figure 1:* 

<img width="635" height="474" alt="Screenshot 2026-01-26 at 18 38 51" src="https://github.com/user-attachments/assets/217fed00-04a8-4acb-b68a-2b6276bfc49f" />

*Figure 2:*
