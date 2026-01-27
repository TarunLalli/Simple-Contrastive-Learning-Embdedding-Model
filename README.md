# Simple Contrastive Learning for Text Embeddings (SimCLR-style)

## Introduction
Here we apply SimCLR architecture to a text based task using random word dropout to create randomised views of data instances to form an array of positive and negative samples for our contrastive task. The architecture follows closely with that using by (Chen, et al., 2020). This work acts as a foundational project towards understanding Contrastive Learning Techniques, in this case applied within the context of NLP. Evaluation of the work consists of a visual analysis of the embedding space using PCA. To accomplish this we implemented a SimCLR-style contrastive framework from scratch, with a custom NT-Xent loss, text based data augmentation, and an encoder-projection architecture.

## Method
The following architectural changes were made in order to accommodate text data:
  * A text encoder is used instead of a ResNet for f(*), encoder
  * Semantic preserving text augmentations are used instead, in this case random word dropout with a probability p=0.2
  * Mean pooling was used before passing embeddings to the projection head, g(*)

Training is done for one epoch of the training_data subset of the hugging face wikitext-2 dataset. Due to the exploratory nature of the study further epochs were not required but form the basis for further study.

## Evaluation
We evaluate the performance of the contrastive learning model using PCA Component visualisation for embedding values for the first 500 data instances of the wikitext-2 training data subset. PCA is used to inspect the relative geometry not the absolute performance. As the trained encoder is not used for a downstream task there was no requirement for quantitative analysis of the model. Plots for both the embedding and projection head outputs were done using the first two principal components for each respectively. These plots are shown in figures 1 and 2.


<img width="317" height="237" alt="Screenshot 2026-01-26 at 18 38 32" src="https://github.com/user-attachments/assets/777371ac-25f0-4af7-98a1-bd87c31eb4d2" />

*Figure 1: scatter plot of first two principle components for PCA of embedding values.* 

<img width="317" height="237" alt="Screenshot 2026-01-26 at 18 38 51" src="https://github.com/user-attachments/assets/217fed00-04a8-4acb-b68a-2b6276bfc49f" />

*Figure 2: scatter plot of first two principle components for PCA of projection head outputs.*

Figure 1 shows two clear clusters, with seperation primarily in PC1, this may be corresponding to sentence length, padding interations in mean pooling, lexical density. This shows successful contrastive learning in the embedding space, were the contrastive loss task to fail we would expect to see a single circular cluster consisting of randomly positioned points. Due to the number of points it is difficult to visualise pairs (1000 pairs for 500 data instances), however, figure 1 appears to show well clustered pairs therefor indicating invariance to data augmentation. Some points can be identified here as outliers, these are most likely due to heavy or complete dropout, a future point of study would be to further investigate the prevelance of outliers across a larger number of epochs and varied dropout probabilities.
 Figure 2 shows a singular tighter cluster than seen in Figure 1, with pairs falling towards each other. We again see outliers here, most likely explained by the same reasons as in Figure 1. We again see inidcation of successful contrastive learning. Figure 2 shows considerabley less noise variance and stronger view alignment over Figure 1. This shows that the projection head is successuly absorbing the contrative loss and protecting the representation space. 

 ## Conclusion
 This project successfully implemented a custom NT-Xent loss function for a contrastive learning task applied to token embeddings. Results show expected geometry, in both the embedding and projection head spaces, for early stage contrastive learning. The nature of the project called for only a brief analysis and training period. Further investigation would therefor, build upon the training and evaluation of the model with the implementation of more typical training, test, and evaluation phases. Additionally the performance of a downstream task would be an excellent investigation into the value of the learned embedding space geometry. These results are consistent with prior findings in contrastive representation learning and validate the implementation.
