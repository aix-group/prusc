# Spurious-free Subnetworks
This is official code for reproducing paper **Out of Spuriousity: Improving Robustness to Spurious Correlations without Group Annotations**

## Abstract
Machine learning models are known to learn spurious correlations, i.e., features having strong correlations with class labels but no causal relation. Relying on those correlations leads to poor performance in the data groups without these correlations and poor generalization ability. To improve the robustness of machine learning models to spurious correlations, we propose an approach to extract a subnetwork from a fully trained network that does not rely on spurious correlations. We observe that spurious correlations induce clusters in the representation space when training with ERM, i.e., data points with the same spurious attribute are close to each other. Based on this observation, we employ a supervised contrastive loss in a novel way to extract a subnetwork that distorts such clusters, forcing the model to unlearn spurious connections. The increase in worst-group accuracy of our approach shows that there exists a subnetwork in a fully trained dense network that is responsible for using only invariant features in classification tasks, therefore erasing the influence of spurious features.

## PruSC
### Prerequisites
- python 3.10
- CUDA 11.6

### Dataset
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [Waterbirds](http://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/)
- [Skin Cancer ISIC](https://www.isic-archive.com)

### Run experiments
Given trained model, run file `cluster/clustering.py` cluster representation space
To run a demo of **PruSC** with CelebA dataset (given ERM representation clusters stored in `demo/celeba_clustering.csv`)

`python3 run.py --mode prune --data celebA --train_root_dir=<DATA_DIR> --imagenet --lambda_sparse 5e-8 --pruning_ep=<EPOCH> --retrain_ep=<EPOCH>`

### Reference
Partial of our code is adapted from two papers [Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks, ICLR 2021](https://github.com/RobertCsordas/modules) and [Training Debiased Subnetworks with Contrastive Weight Pruning, CVPR 2023](https://github.com/geonyeong-park/DCWP)
