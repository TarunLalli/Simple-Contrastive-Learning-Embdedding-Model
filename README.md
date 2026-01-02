# Simple-Contrastive-Learning-Embdedding-Model

Simple contrastive embedding learning model applied to text embedding task

In this project, we apply the SimCLR architecture to a text based task. The general architecture is kept consistent, with the following changes made:
  * A text encoder is used instead of a ResNet for f(*)
  * Semantic preserving text augmentations are used instead

This study acts as an exploration into the affect of contrastive learning on the embdedding space. Evaluation of the work consists of a visual analysis of the embedding space using t-SNE. Note, in order to optimise training times and pytorch architecture we implement a preconfigured contrastive learning loss instead of the one built earlier.
