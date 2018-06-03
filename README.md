# Facial-Recognition-Via-Sparse-Recognition
Neural Networks are fun, but here's a way to recognize people using Compressed Sensing methods

This package does not contain the original AR face data files, which need to be obtained through the link : http://www2.ece.ohio-state.edu/~aleix/ARdatabase.html.

Sincere apologies for the presentation and article summary as they are written in French... I hope to translate that soon. In the meantime, the Jupyter Notebook is fully commented in English.

NOTE : In the feature_reduction.py, there's an argument "nb_component". It comes from the idea that we don't need every information available in a given picture. It creates a Matrice so large at the end, that computing becomes tedious. Therefore it's nice to reduce the size of the matrices, that's when nb_component comes in. It's a number which you can change and it corresponds to the number of pixels you would like to consider, as it becomes a dimension factor. For example, original images are 120x165, dimension would then be 19800, whereas resized pictures that are size 30x42 yields 1260, which is sufficient as we show in our presentation.
