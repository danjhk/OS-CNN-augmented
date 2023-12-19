### Time series augmentation for the Omni-scale block: towards robust time series classification

This code is an improvement for the paper [Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification (ICLR 2022)](https://arxiv.org/abs/2002.10061) in which we apply time series augmentation techniques on the UCR Time Series Classification Archive 2018.

### Environment 

python == 3.10.12 \
pytorch == 2.1.1 \
scikit-learn == 1.3.2\

### Reproducing the results

First download the UCR Time Series Classification Archive 2018 and then run the OS-CNN-data-augmentation.py script. \
Then, run the post_processor.ipynb to get the results from the tables 2 to 5 of the paper. \
To reproduce the transformations in Figure 1 of the paper, run the code in post_processor.ipynb under the section "Visualizing the transformations".


### Saved Results

The folder acc_results contains the accuracies obtained by using only transformed data ("all_accs.csv") and augmented data ("aug_accs.csv"). \
The folder aug_results contains intermediate results that we computed before obtaining the results in all_accs.csv and aug_accs.csv and in the tables 2 to 5 of the paper.