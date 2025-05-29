#### SNIP: SINGLE -SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY

Namhoon Lee, Thalaiyasingam Ajanthan & Philip H. S. Torr University of Oxford

{namhoon,ajanthan,phst }@robots.ox.ac.uk

### ABSTRACT

Pruning large neural networks while maintaining their performance is often desirable due to the reduced space and time complexity. In existing methods, pruning is done within an iterative optimization procedure with either heuristically designed pruning schedules or additional hyperparameters, undermining their utility. In this work, we present a new approach that prunes a given network once at initialization prior to training. To achieve this, we introduce a saliency criterion based on connection sensitivity that identifies structurally important connections in the network for the given task. This eliminates the need for both pretraining and the complex pruning schedule while making it robust to architecture variations. After pruning, the sparse network is trained in the standard way. Our method obtains extremely sparse networks with virtually the same accuracy as the reference network on the MNIST , CIFAR-10, and Tiny-ImageNet classification tasks and is broadly applicable to various architectures including convolutional, residual and recurrent networks. Unlike existing methods, our approach enables us to demonstrate that the retained connections are indeed relevant to the given task.

### 1 INTRODUCTION

Despite the success of deep neural networks in machine learning, they are often found to be highly overparametrized making them computationally expensive with excessive memory requirements. Pruning such large networks with minimal loss in performance is appealing for real-time applications, especially on resource-limited devices. In addition, compressed neural networks utilize the model capacity efficiently, and this interpretation can be used to derive better generalization bounds for neural networks [\(Arora et al.](#page-10-0) [\(2018\)](#page-10-0)).

In network pruning, given a large reference neural network, the goal is to learn a much smaller subnetwork that mimics the performance of the reference network. The majority of existing methods in the literature attempt to find a subset of weights from the pretrained reference network either based on a saliency criterion [\(Mozer & Smolensky \(1989](#page-11-0)); [LeCun et al.](#page-11-1) [\(1990\)](#page-11-1); [Han et al. \(2015\)](#page-10-1)) or utilizing sparsity enforcing penalties [\(Chauvin \(1989\)](#page-10-2); [Carreira-Perpi ˜n´an & Idelbayev \(2018\)](#page-10-3)). Unfortunately, since pruning is included as a part of an iterative optimization procedure, all these methods require many expensive *prune – retrain cycles* and heuristic design choices with additional hyperparameters, making them non-trivial to extend to new architectures and tasks.

In this work, we introduce a saliency criterion that identifies connections in the network that are important to the given task in a data-dependent way before training. Specifically, we discover important connections based on their influence on the loss function at a variance scaling initialization, which we call connection sensitivity. Given the desired sparsity level, redundant connections are pruned once prior to training (*i.e*., single-shot), and then the sparse pruned network is trained in the standard way. Our approach has several attractive properties:

- *Simplicity.* Since the network is pruned once prior to training, there is no need for pretraining and complex pruning schedules. Our method has no additional hyperparameters and once pruned, training of the sparse network is performed in the standard way.
- *Versatility.* Since our saliency criterion chooses structurally important connections, it is robust to architecture variations. Therefore our method can be applied to various architectures including convolutional, residual and recurrent networks with no modifications.

• *Interpretability.* Our method determines important connections with a mini-batch of data at single-shot. By varying this mini-batch used for pruning, our method enables us to verify that the retained connections are indeed essential for the given task.

We evaluate our method on MNIST, CIFAR-10, and Tiny-ImageNet classification datasets with widely varying architectures. Despite being the simplest, our method obtains extremely sparse networks with virtually the same accuracy as the existing baselines across all tested architectures. Furthermore, we investigate the relevance of the retained connections as well as the effect of the network initialization and the dataset on the saliency score.

# 2 RELATED WORK

Classical methods. Essentially, early works in network pruning can be categorized into two groups [\(Reed \(1993\)](#page-11-2)): 1) those that utilize sparsity enforcing penalties; and 2) methods that prune the network based on some saliency criterion. The methods from the former category [\(Chauvin](#page-10-2) [\(1989\)](#page-10-2); [Weigend et al. \(1991\)](#page-11-3); [Ishikawa \(1996\)](#page-10-4)) augment the loss function with some sparsity enforcing penalty terms (*e.g*., L<sup>0</sup> or L<sup>1</sup> norm), so that back-propagation effectively penalizes the magnitude of the weights during training. Then weights below a certain threshold may be removed. On the other hand, classical saliency criteria include the sensitivity of the loss with respect to the neurons [\(Mozer & Smolensky](#page-11-0) [\(1989\)](#page-11-0)) or the weights [\(Karnin \(1990\)](#page-10-5)) and Hessian of the loss with respect to the weights [\(LeCun et al. \(1990](#page-11-1)); [Hassibi et al.](#page-10-6) [\(1993\)](#page-10-6)). Since these criteria are heavily dependent on the scale of the weights and are designed to be incorporated within the learning process, these methods are prohibitively slow requiring many iterations of pruning and learning steps. Our approach identifies redundant weights from an architectural point of view and prunes them once at the beginning before training.

Modern advances. In recent years, the increased space and time complexities as well as the risk of overfitting in deep neural networks prompted a surge of further investigation in network pruning. While Hessian based approaches employ the diagonal approximation due to its computational simplicity, impressive results (*i.e*., extreme sparsity without loss in accuracy) are achieved using magnitude of the weights as the criterion [\(Han et al. \(2015](#page-10-1))). This made them the de facto standard method for network pruning and led to various implementations [\(Guo et al. \(2016\)](#page-10-7); [Carreira-Perpi ˜n´an & Idelbayev \(2018\)](#page-10-3)). The magnitude criterion is also extended to recurrent neural networks [\(Narang et al.](#page-11-4) [\(2017\)](#page-11-4)), yet with heavily tuned hyperparameter setting. Unlike our approach, the main drawbacks of magnitude based approaches are the reliance on pretraining and the expensive prune – retrain cycles. Furthermore, since pruning and learning steps are intertwined, they often require highly heuristic design choices which make them non-trivial to be extended to new architectures and different tasks. Meanwhile, Bayesian methods are also applied to network pruning [\(Ullrich et al.](#page-11-5) [\(2017\)](#page-11-5); [Molchanov et al. \(2017a\)](#page-11-6)) where the former extends the soft weight sharing in [Nowlan & Hinton \(1992](#page-11-7)) to obtain a sparse and compressed network, and the latter uses variational inference to learn the dropout rate which can then be used to prune the network. Unlike the above methods, our approach is simple and easily adaptable to any given architecture or task without modifying the pruning procedure.

Network compression in general. Apart from weight pruning, there are approaches focused on structured simplification such as pruning filters [\(Li et al. \(2017\)](#page-11-8); [Molchanov et al. \(2017b\)](#page-11-9)), structured sparsity with regularizers [\(Wen et al. \(2016\)](#page-11-10)), low-rank approximation [\(Jaderberg et al.](#page-10-8) [\(2014\)](#page-10-8)), matrix and tensor factorization [\(Novikov et al. \(2015\)](#page-11-11)), and sparsification using expander graphs [\(Prabhu et al.](#page-11-12) [\(2018\)](#page-11-12)) or Erd ˝os-R´enyi random graph [\(Mocanu et al.](#page-11-13) [\(2018\)](#page-11-13)). In addition, there is a large body of work on compressing the representation of weights. A non-exhaustive list includes quantization [\(Gong et al. \(2014\)](#page-10-9)), reduced precision [\(Gupta et al.](#page-10-10) [\(2015\)](#page-10-10)) and binary weights [\(Hubara et al.](#page-10-11) [\(2016\)](#page-10-11)). In this work, we focus on weight pruning that is free from structural constraints and amenable to further compression schemes.

# <span id="page-1-0"></span>3 NEURAL NETWORK PRUNING

The main hypothesis behind the neural network pruning literature is that neural networks are usually overparametrized, and comparable performance can be obtained by a much smaller network [\(Reed](#page-11-2) [\(1993\)](#page-11-2)) while improving generalization [\(Arora et al.](#page-10-0) [\(2018\)](#page-10-0)). To this end, the objective is to learn a sparse network while maintaining the accuracy of the standard reference network. Let us first formulate neural network pruning as an optimization problem.

Given a dataset D = {(x<sup>i</sup> , yi)} n <sup>i</sup>=1, and a desired sparsity level κ (*i.e*., the number of non-zero weights) neural network pruning can be written as the following constrained optimization problem:

<span id="page-2-0"></span>
$$\min\_{\mathbf{w}} L(\mathbf{w}; \mathcal{D}) = \min\_{\mathbf{w}} \frac{1}{n} \sum\_{i=1}^{n} \ell(\mathbf{w}; (\mathbf{x}\_i, \mathbf{y}\_i))\,,\tag{1}$$
 
$$\text{s.t.} \quad \mathbf{w} \in \mathbb{R}^m, \quad ||\mathbf{w}||\_0 \le \kappa\,\,.$$

Here, ℓ(·) is the standard loss function (*e.g*., cross-entropy loss), w is the set of parameters of the neural network, m is the total number of parameters and k · k<sup>0</sup> is the standard L<sup>0</sup> norm.

The conventional approach to optimize the above problem is by adding sparsity enforcing penalty terms [\(Chauvin](#page-10-2) [\(1989\)](#page-10-2); [Weigend et al. \(1991](#page-11-3)); [Ishikawa](#page-10-4) [\(1996\)](#page-10-4)). Recently, [Carreira-Perpi ˜n´an & Idelbayev \(2018\)](#page-10-3) attempts to minimize the above constrained optimization problem using the stochastic version of projected gradient descent (where the projection is accomplished by pruning). However, these methods often turn out to be inferior to saliency based methods in terms of resulting sparsity and require heavily tuned hyperparameter settings to obtain comparable results.

On the other hand, saliency based methods treat the above problem as selectively removing redundant parameters (or connections) in the neural network. In order to do so, one has to come up with a good criterion to identify such redundant connections. Popular criteria include magnitude of the weights, *i.e*., weights below a certain threshold are redundant [\(Han et al.](#page-10-1) [\(2015\)](#page-10-1); [Guo et al. \(2016\)](#page-10-7)) and Hessian of the loss with respect to the weights, *i.e*., the higher the value of Hessian, the higher the importance of the parameters [\(LeCun et al.](#page-11-1) [\(1990\)](#page-11-1); [Hassibi et al.](#page-10-6) [\(1993\)](#page-10-6)), defined as follows:

$$s\_j = \begin{cases} \left| w\_j \right|, & \text{for magnitude based} \\ \frac{w\_j^2 H\_{jj}}{2} & \text{or } \frac{w\_j^2}{2H\_{jj}^{-1}} & \text{for Hessian based }. \end{cases} \tag{2}$$

Here, for connection j, s<sup>j</sup> is the saliency score, w<sup>j</sup> is the weight, and Hjj is the value of the Hessian matrix, where the Hessian H = ∂ <sup>2</sup>L/∂w<sup>2</sup> ∈ R <sup>m</sup>×<sup>m</sup>. Considering Hessian based methods, the Hessian matrix is neither diagonal nor positive definite in general, approximate at best, and intractable to compute for large networks.

Despite being popular, both of these criteria depend on the scale of the weights and in turn require pretraining and are very sensitive to the architectural choices. For instance, different normalization layers affect the scale of the weights in a different way, and this would non-trivially affect the saliency score. Furthermore, pruning and the optimization steps are alternated many times throughout training, resulting in highly expensive *prune – retrain cycles*. Such an exorbitant requirement hinders the use of pruning methods in large-scale applications and raises questions about the credibility of the existing pruning criteria.

In this work, we design a criterion which directly measures the connection importance in a datadependent manner. This alleviates the dependency on the weights and enables us to prune the network once at the beginning, and then the training can be performed on the sparse pruned network. Therefore, our method eliminates the need for the expensive prune – retrain cycles, and in theory, it can be an order of magnitude faster than the standard neural network training as it can be implemented using software libraries that support sparse matrix computations.

### 4 SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY

Given a neural network and a dataset, our goal is to design a method that can selectively prune redundant connections for the given task in a data-dependent way even before training. To this end, we first introduce a criterion to identify important connections and then discuss its benefits.

#### 4.1 CONNECTION SENSITIVITY: ARCHITECTURAL PERSPECTIVE

Since we intend to measure the importance (or sensitivity) of each connection independently of its weight, we introduce auxiliary indicator variables c ∈ {0, 1} <sup>m</sup> representing the connectivity of parameters w. [1](#page-3-0) Now, given the sparsity level κ, Equation [1](#page-2-0) can be correspondingly modified as:

$$\begin{aligned} \min\_{\mathbf{c}, \mathbf{w}} L(\mathbf{c} \odot \mathbf{w}; \mathcal{D}) &= \min\_{\mathbf{c}, \mathbf{w}} \frac{1}{n} \sum\_{i=1}^{n} \ell(\mathbf{c} \odot \mathbf{w}; (\mathbf{x}\_i, \mathbf{y}\_i)) \,, \\ \text{s.t.} \quad & \mathbf{w} \in \mathbb{R}^m \,, \\ & \mathbf{c} \in \{0, 1\}^m, \quad \|\mathbf{c}\|\_0 \le \kappa \,, \end{aligned} \tag{3}$$

where ⊙ denotes the Hadamard product. Compared to Equation [1,](#page-2-0) we have doubled the number of learnable parameters in the network and directly optimizing the above problem is even more difficult. However, the idea here is that since we have separated the weight of the connection (w) from whether the connection is present or not (c), we may be able to determine the importance of each connection by measuring its effect on the loss function.

For instance, the value of c<sup>j</sup> indicates whether the connection j is active (c<sup>j</sup> = 1) in the network or pruned (c<sup>j</sup> = 0). Therefore, to measure the effect of connection j on the loss, one can try to measure the difference in loss when c<sup>j</sup> = 1 and c<sup>j</sup> = 0, keeping everything else constant. Precisely, the effect of removing connection j can be measured by,

$$
\Delta L\_j(\mathbf{w}; \mathcal{D}) = L(\mathbf{1} \odot \mathbf{w}; \mathcal{D}) - L((\mathbf{1} - \mathbf{e}\_j) \odot \mathbf{w}; \mathcal{D}) \,, \tag{4}
$$

where e<sup>j</sup> is the indicator vector of element j (*i.e*., zeros everywhere except at the index j where it is one) and 1 is the vector of dimension m.

Note that computing ∆L<sup>j</sup> for each j ∈ {1 . . . m} is prohibitively expensive as it requires m + 1 (usually in the order of millions) forward passes over the dataset. In fact, since c is binary, L is not differentiable with respect to c, and it is easy to see that ∆L<sup>j</sup> attempts to measure the influence of connection j on the loss function in this discrete setting. Therefore, by relaxing the binary constraint on the indicator variables c, ∆L<sup>j</sup> can be approximated by the derivative of L with respect to c<sup>j</sup> , which we denote g<sup>j</sup> (w; D). Hence, the effect of connection j on the loss can be written as:

$$\Delta L\_j(\mathbf{w}; \mathcal{D}) \approx g\_j(\mathbf{w}; \mathcal{D}) = \left. \frac{\partial L(\mathbf{c} \odot \mathbf{w}; \mathcal{D})}{\partial c\_j} \right|\_{\mathbf{c} = \mathbf{1}} = \lim\_{\delta \to 0} \left. \frac{L(\mathbf{c} \odot \mathbf{w}; \mathcal{D}) - L((\mathbf{c} - \delta \mathbf{e}\_j) \odot \mathbf{w}; \mathcal{D})}{\delta} \right|\_{\mathbf{c} = \mathbf{1}} \tag{5}$$

In fact, ∂L/∂c<sup>j</sup> is an infinitesimal version of ∆Lj, that measures the rate of change of L with respect to an infinitesimal change in c<sup>j</sup> from 1 → 1 − δ. This can be computed efficiently in one forward-backward pass using automatic differentiation, for all j at once. Notice, this formulation can be viewed as perturbing the weight w<sup>j</sup> by a multiplicative factor δ and measuring the change in loss. This approximation is similar in spirit to [Koh & Liang \(2017\)](#page-10-12) where they try to measure the influence of a datapoint to the loss function. Here we measure the influence of connections. Furthermore, ∂L/∂c<sup>j</sup> is not to be confused with the gradient with respect to the weights (∂L/∂w<sup>j</sup> ), where the change in loss is measured with respect to an additive change in weight w<sup>j</sup> .

Notably, our interest is to discover important (or sensitive) connections in the architecture, so that we can prune unimportant ones in single-shot, disentangling the pruning process from the iterative optimization cycles. To this end, we take the magnitude of the derivatives g<sup>j</sup> as the saliency criterion. Note that if the magnitude of the derivative is high (regardless of the sign), it essentially means that the connection c<sup>j</sup> has a considerable effect on the loss (either positive or negative), and it has to be preserved to allow learning on w<sup>j</sup> . Based on this hypothesis, we define connection sensitivity as the normalized magnitude of the derivatives:

<span id="page-3-1"></span>
$$s\_j = \frac{|g\_j(\mathbf{w}; \mathcal{D})|}{\sum\_{k=1}^{m} |g\_k(\mathbf{w}; \mathcal{D})|} \,. \tag{6}$$

<span id="page-3-2"></span>.

Once the sensitivity is computed, only the top-κ connections are retained, where κ denotes the desired number of non-zero weights. Precisely, the indicator variables c are set as follows:

$$c\_j = \mathbb{1}[s\_j - \tilde{s}\_\kappa \ge 0] \,, \quad \forall j \in \{1 \ldots m\} \,, \tag{7}$$

where s˜<sup>κ</sup> is the κ-th largest element in the vector s and <sup>1</sup>[·] is the indicator function. Here, for exactly κ connections to be retained, ties can be broken arbitrarily.

We would like to clarify that the above criterion (Equation [6\)](#page-3-1) is different from the criteria used in early works by [Mozer & Smolensky](#page-11-0) [\(1989\)](#page-11-0) or [Karnin \(1990](#page-10-5)) which do not entirely capture the

<span id="page-3-0"></span><sup>1</sup>Multiplicative coefficients (similar to c) were also used for subset regression in [Breiman \(1995](#page-10-13)).

| Algorithm 1 SNIP: Single-shot Network Pruning based on Connection Sensitivity |                                        |  |  |  |  |
|-------------------------------------------------------------------------------|----------------------------------------|--|--|--|--|
| Require: Loss function L, training dataset D, sparsity level κ                | ⊲ Refer Equation 3                     |  |  |  |  |
| Ensure: kw∗k0<br>≤ κ                                                          |                                        |  |  |  |  |
| 1: w ← VarianceScalingInitialization                                          | ⊲ Refer Section 4.2                    |  |  |  |  |
| 2: Db =<br>b<br>{(xi<br>, yi)}<br>i=1 ∼ D                                     | ⊲ Sample a mini-batch of training data |  |  |  |  |
| gj (w;Db<br>) <br>3: sj<br>←<br>,<br>∀j ∈ {1 m}<br>m<br>k=1 gk(w;Db) <br>P    | ⊲ Connection sensitivity               |  |  |  |  |
| 4: ˜s ← SortDescending(s)                                                     |                                        |  |  |  |  |
| 5: cj<br>← 1[sj<br>− s˜κ<br>≥ 0] ,<br>∀ j ∈ {1m}                              | ⊲ Pruning: choose top-κ connections    |  |  |  |  |
| 6: w∗ ←<br>L(c ⊙ w; D)<br>arg minw∈Rm                                         | ⊲ Regular training                     |  |  |  |  |
| 7: w∗ ←<br>c ⊙ w∗                                                             |                                        |  |  |  |  |

<span id="page-4-1"></span>

connection sensitivity. The fundamental idea behind them is to identify elements (*e.g*. weights or neurons) that least degrade the performance when removed. This means that their saliency criteria (*i.e*. −∂L/∂w or −∂L/∂α; α refers to the connectivity of neurons), in fact, depend on the loss value before pruning, which in turn, require the network to be pre-trained and iterative optimization cycles to ensure minimal loss in performance. They also suffer from the same drawbacks as the magnitude and Hessian based methods as discussed in Section [3.](#page-1-0) In contrast, our saliency criterion (Equation [6\)](#page-3-1) is designed to measure the sensitivity as to how much influence elements have on the loss function regardless of whether it is positive or negative. This criterion alleviates the dependency on the value of the loss, eliminating the need for pre-training. These fundamental differences enable the network to be pruned at single-shot prior to training, which we discuss further in the next section.

### <span id="page-4-0"></span>4.2 SINGLE-SHOT PRUNING AT INITIALIZATION

Note that the saliency measure defined in Equation [6](#page-3-1) depends on the value of weights w used to evaluate the derivative as well as the dataset D and the loss function L. In this section, we discuss the effect of each of them and show that it can be used to prune the network in single-shot with initial weights w.

Firstly, in order to minimize the impact of weights on the derivatives ∂L/∂cj, we need to choose these weights carefully. For instance, if the weights are too large, the activations after the non-linear function (*e.g*., sigmoid) will be saturated, which would result in uninformative gradients. Therefore, the weights should be within a sensible range. In particular, there is a body of work on neural network initialization [\(Goodfellow et al. \(2016\)](#page-10-14)) that ensures the gradients to be in a reasonable range, and our saliency measure can be used to prune neural networks at any such initialization.

Furthermore, we are interested in making our saliency measure robust to architecture variations. Note that initializing neural networks is a random process, typically done using normal distribution. However, if the initial weights have a fixed variance, the signal passing through each layer no longer guarantees to have the same variance, as noted by [LeCun et al.](#page-11-14) [\(1998\)](#page-11-14). This would make the gradient and in turn our saliency measure, to be dependent on the architectural characteristics. Thus, we advocate the use of variance scaling methods (*e.g*., [Glorot & Bengio \(2010](#page-10-15))) to initialize the weights, such that the variance remains the same throughout the network. By ensuring this, we empirically show that our saliency measure computed at initialization is robust to variations in the architecture.

Next, since the dataset and the loss function defines the task at hand, by relying on both of them, our saliency criterion in fact discovers the connections in the network that are important to the given task. However, the practitioner needs to make a choice on whether to use the whole training set, or a mini-batch or the validation set to compute the connection saliency. Moreover, in case there are memory limitations (*e.g*., large model or dataset), one can accumulate the saliency measure over multiple batches or take an exponential moving average. In our experiments, we show that using only one mini-batch of a reasonable number of training examples can lead to effective pruning.

Finally, in contrast to the previous approaches, our criterion for finding redundant connections is simple and directly based on the sensitivity of the connections. This allows us to effectively identify and prune redundant connections in a single step even before training. Then, training can be performed on the resulting pruned (sparse) network. We name our method SNIP for Single-shot Network Pruning, and the complete algorithm is given in Algorithm [1.](#page-4-1)

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 1: Test errors of LeNets pruned at varying sparsity levels κ¯, where κ¯ = 0 refers to the reference network trained without pruning. Our approach performs as good as the reference network across varying sparsity levels on both the models.

# 5 EXPERIMENTS

We evaluate our method, SNIP, on MNIST, CIFAR-10 and Tiny-ImageNet classification tasks with a variety of network architectures. Our results show that SNIP yields extremely sparse models with minimal or no loss in accuracy across all tested architectures, while being much simpler than other state-of-the-art alternatives. We also provide clear evidence that our method prunes genuinely explainable connections rather than performing blind pruning.

Experiment setup For brevity, we define the sparsity level to be κ¯ = (m − κ)/m · 100 (%), where m is the total number of parameters and κ is the desired number of non-zero weights. For a given sparsity level κ¯, the sensitivity scores are computed using a batch of 100 and 128 examples for MNIST and CIFAR experiments, respectively. After pruning, the pruned network is trained in the standard way. Specifically, we train the models using SGD with momentum of 0.9, batch size of 100 for MNIST and 128 for CIFAR experiments and the weight decay rate of 0.0005, unless stated otherwise. The initial learning rate is set to 0.1 and decayed by 0.1 at every 25k or 30k iterations for MNIST and CIFAR, respectively. Our algorithm requires no other hyperparameters or complex learning/pruning schedules as in most pruning algorithms. We spare 10% of the training data as a validation set and used only 90% for training. For CIFAR experiments, we use the standard data augmentation (*i.e*., random horizontal flip and translation up to 4 pixels) for both the reference and sparse models. The code can be found here: [https://github.com/namhoonlee/snip-public.](https://github.com/namhoonlee/snip-public)

### 5.1 PRUNING LENETS WITH VARYING LEVELS OF SPARSITY

We first test our approach on two standard networks for pruning, LeNet-300-100 and LeNet-5-Caffe. LeNet-300-100 consists of three fully-connected (fc) layers with 267k parameters and LeNet-5-Caffe consists of two convolutional (conv) layers and two fc layers with 431k parameters. We prune the LeNets for different sparsity levels κ¯ and report the performance in error on the MNIST image classification task. We run the experiment 20 times for each κ¯ by changing random seeds for dataset and network initialization. The results are reported in Figure [1.](#page-5-0)

The pruned sparse LeNet-300-100 achieves performances similar to the reference (κ¯ = 0), only with negligible loss at κ¯ = 90. For LeNet-5-Caffe, the performance degradation is nearly invisible. Note that our saliency measure does not require the network to be pre-trained and is computed at random initialization. Despite such simplicity, our approach prunes LeNets quickly (single-shot) and effectively (minimal accuracy loss) at varying sparsity levels.

### 5.2 COMPARISONS TO EXISTING APPROACHES

What happens if we increase the target sparsity to an extreme level? For example, would a model with only 1% of the total parameters still be trainable and perform well? We test our approach for extreme sparsity levels (*e.g*., up to 99% sparsity on LeNet-5-Caffe) and compare with various pruning algorithms as follows: LWC [\(Han et al. \(2015\)](#page-10-1)), DNS [\(Guo et al. \(2016\)](#page-10-7)), LC [\(Carreira-Perpi ˜n´an & Idelbayev \(2018\)](#page-10-3)), SWS [\(Ullrich et al. \(2017\)](#page-11-5)), SVD [\(Molchanov et al.](#page-11-6)

<span id="page-6-0"></span>

| Method      | Criterion                 | κ¯ (%)       | LeNet-300-100<br>err. (%) | κ¯ (%)       | LeNet-5-Caffe<br>err. (%) | Pretrain | # Prune | Additional<br>hyperparam. objective constraints | Augment | Arch. |
|-------------|---------------------------|--------------|---------------------------|--------------|---------------------------|----------|---------|-------------------------------------------------|---------|-------|
| Ref.        | –                         | –            | 1.7                       | –            | 0.9                       | –        | –       | –                                               | –       | –     |
| LWC         | Magnitude                 | 91.7         | 1.6                       | 91.7         | 0.8                       | X        | many    | X                                               | ✗       | X     |
| DNS         | Magnitude                 | 98.2         | 2.0                       | 99.1         | 0.9                       | X        | many    | X                                               | ✗       | X     |
| LC          | Magnitude                 | 99.0         | 3.2                       | 99.0         | 1.1                       | X        | many    | X                                               | X       | ✗     |
| SWS         | Bayesian                  | 95.6         | 1.9                       | 99.5         | 1.0                       | X        | soft    | X                                               | X       | ✗     |
| SVD         | Bayesian                  | 98.5         | 1.9                       | 99.6         | 0.8                       | X        | soft    | X                                               | X       | ✗     |
| OBD         | Hessian                   | 92.0         | 2.0                       | 92.0         | 2.7                       | X        | many    | X                                               | ✗       | ✗     |
| L-OBS       | Hessian                   | 98.5         | 2.0                       | 99.0         | 2.1                       | X        | many    | X                                               | ✗       | X     |
| SNIP (ours) | Connection<br>sensitivity | 95.0<br>98.0 | 1.6<br>2.4                | 98.0<br>99.0 | 0.8<br>1.1                | ✗        | 1       | ✗                                               | ✗       | ✗     |

Table 1: Pruning results on LeNets and comparisons to other approaches. Here, "many" refers to an arbitrary number often in the order of total learning steps, and "soft" refers to soft pruning in Bayesian based methods. Our approach is capable of pruning up to 98% for LeNet-300-100 and 99% for LeNet-5-Caffe with marginal increases in error from the reference network. Notably, our approach is considerably simpler than other approaches, with no requirements such as pretraining, additional hyperparameters, augmented training objective or architecture dependent constraints.

[\(2017a\)](#page-11-6)), OBD [\(LeCun et al.](#page-11-1) [\(1990\)](#page-11-1)), L-OBS [\(Dong et al.](#page-10-16) [\(2017\)](#page-10-16)). The results are summarized in Table [1.](#page-6-0)

We achieve errors that are comparable to the reference model, degrading approximately 0.7% and 0.3% while pruning 98% and 99% of the parameters in LeNet-300-100 and LeNet-5-Caffe respectively. For slightly relaxed sparsities (*i.e*., 95% for LeNet-300-100 and 98% for LeNet-5-Caffe), the sparse models pruned by SNIP record better performances than the dense reference network. Considering 99% sparsity, our method efficiently finds 1% of the connections even before training, that are sufficient to learn as good as the reference network. Moreover, SNIP is competitive to other methods, yet it is unparalleled in terms of algorithm simplicity.

To be more specific, we enumerate some key points and non-trivial aspects of other algorithms and highlight the benefit of our approach. First of all, the aforementioned methods require networks to be fully trained (if not partly) before pruning. These approaches typically perform many pruning operations even if the network is well pretrained, and require additional hyperparameters (*e.g*., pruning frequency in [Guo et al. \(2016\)](#page-10-7), annealing schedule in [Carreira-Perpi ˜n´an & Idelbayev \(2018](#page-10-3))). Some methods augment the training objective to handle pruning together with training, increasing the complexity of the algorithm (*e.g*., augmented Lagrangian in [Carreira-Perpi ˜n´an & Idelbayev \(2018](#page-10-3)), variational inference in [Molchanov et al. \(2017a\)](#page-11-6)). Furthermore, there are approaches designed to include architecture dependent constraints (*e.g*., layer-wise pruning schemes in [Dong et al.](#page-10-16) [\(2017\)](#page-10-16)).

Compared to the above approaches, ours seems to cost almost nothing; it requires no pretraining or additional hyperparameters, and is applied only once at initialization. This means that one can easily plug-in SNIP as a preprocessor before training neural networks. Since SNIP prunes the network at the beginning, we could potentially expedite the training phase by training only the survived parameters (*e.g*., reduced expected FLOPs in [Louizos et al.](#page-11-15) [\(2018\)](#page-11-15)). Notice that this is not possible for the aforementioned approaches as they obtain the maximum sparsity at the end of the process.

#### 5.3 VARIOUS MODERN ARCHITECTURES

In this section we show that our approach is generally applicable to more complex modern network architectures including deep convolutional, residual and recurrent ones. Specifically, our method is applied to the following models:

- AlexNet-s and AlexNet-b: Models similar to [Krizhevsky et al. \(2012](#page-11-16)) in terms of the number of layers and size of kernels. We set the size of fc layers to 512 (AlexNet-s) and to 1024 (AlexNet-b) to adapt for CIFAR-10 and use strides of 2 for all conv layers instead of using pooling layers.
- VGG-C, VGG-D and VGG-like: Models similar to the original VGG models described in [Simonyan & Zisserman \(2015\)](#page-11-17). VGG-like [\(Zagoruyko](#page-11-18) [\(2015\)](#page-11-18)) is a popular variant adapted for CIFAR-10 which has one less fc layers. For all VGG models, we set the size of fc layers to 512, remove dropout layers to avoid any effect on sparsification and use batch normalization instead.
- WRN-16-8, WRN-16-10 and WRN-22-8: Same models as in [Zagoruyko & Komodakis \(2016\)](#page-11-19).

<span id="page-7-0"></span>

| Architecture  | Model                                                | Sparsity (%)                         | # Parameters                                                                              | Error (%)                                                                                  | ∆                                         |
|---------------|------------------------------------------------------|--------------------------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------|
| Convolutional | AlexNet-s<br>AlexNet-b<br>VGG-C<br>VGG-D<br>VGG-like | 90.0<br>90.0<br>95.0<br>95.0<br>97.0 | 5.1m →<br>507k<br>8.5m →<br>849k<br>10.5m →<br>526k<br>15.2m →<br>762k<br>15.0m →<br>449k | 14.12 →<br>14.99<br>13.92 →<br>14.50<br>6.82 →<br>7.27<br>6.76 →<br>7.09<br>8.26 →<br>8.00 | +0.87<br>+0.58<br>+0.45<br>+0.33<br>−0.26 |
| Residual      | WRN-16-8<br>WRN-16-10<br>WRN-22-8                    | 95.0<br>95.0<br>95.0                 | 10.0m →<br>548k<br>17.1m →<br>856k<br>17.2m →<br>858k                                     | 6.21 →<br>6.63<br>5.91 →<br>6.43<br>6.14 →<br>5.85                                         | +0.42<br>+0.52<br>−0.29                   |
| Recurrent     | LSTM-s<br>LSTM-b<br>GRU-s<br>GRU-b                   | 95.0<br>95.0<br>95.0<br>95.0         | 137k →<br>6.8k<br>535k →<br>26.8k<br>104k →<br>5.2k<br>404k →<br>20.2k                    | 1.88 →<br>1.57<br>1.15 →<br>1.35<br>1.87 →<br>2.41<br>1.71 →<br>1.52                       | −0.31<br>+0.20<br>+0.54<br>−0.19          |

Table 2: Pruning results of the proposed approach on various modern architectures (before → after). AlexNets, VGGs and WRNs are evaluated on CIFAR-10, and LSTMs and GRUs are evaluated on the sequential MNIST classification task. The approach is generally applicable regardless of architecture types and models and results in a significant amount of reduction in the number of parameters with minimal or no loss in performance.

• LSTM-s, LSTM-b, GRU-s, GRU-b: One layer RNN networks with either LSTM [\(Zaremba et al.](#page-11-20) [\(2014](#page-11-20))) or GRU [\(Cho et al.](#page-10-17) [\(2014\)](#page-10-17)) cells. We develop two unit sizes for each cell type, 128 and 256 for {·}-s and {·}-b, respectively. The model is adapted for the sequential MNIST classification task, similar to [Le et al.](#page-11-21) [\(2015\)](#page-11-21). Instead of processing pixel-by-pixel, however, we perform rowby-row processing (*i.e*., the RNN cell receives each row at a time).

The results are summarized in Table [2.](#page-7-0) Overall, our approach prunes a substantial amount of parameters in a variety of network models with minimal or no loss in accuracy (< 1%). Our pruning procedure does not need to be modified for specific architectural variations (*e.g*., recurrent connections), indicating that it is indeed versatile and scalable. Note that prior art that use a saliency criterion based on the weights (*i.e*., magnitude or Hessian based) would require considerable adjustments in their pruning schedules as per changes in the model.

We note of a few challenges in directly comparing against others: different network specifications, learning policies, datasets and tasks. Nonetheless, we provide a few comparison points that we found in the literature. On CIFAR-10, SVD prunes 97.9% of the connections in VGG-like with no loss in accuracy (ours: 97% sparsity) while SWS obtained 93.4% sparsity on WRN-16-4 but with a non-negligible loss in accuracy of 2%. There are a couple of works attempting to prune RNNs (*e.g*., GRU in [Narang et al.](#page-11-4) [\(2017\)](#page-11-4) and LSTM in [See et al. \(2016\)](#page-11-22)). Even though these methods are specifically designed for RNNs, none of them are able to obtain extreme sparsity without substantial loss in accuracy reflecting the challenges of pruning RNNs. To the best of our knowledge, we are the first to demonstrate on convolutional, residual and recurrent networks for extreme sparsities without requiring additional hyperparameters or modifying the pruning procedure.

### 5.4 UNDERSTANDING WHICH CONNECTIONS ARE BEING PRUNED

So far we have shown that our approach can prune a variety of deep neural network architectures for extreme sparsities without losing much on accuracy. However, it is not clear yet which connections are actually being pruned away or whether we are pruning the right (*i.e*., unimportant) ones. What if we could actually peep through our approach into this inspection?

Consider the first layer in LeNet-300-100 parameterized by wl=1 ∈ R <sup>784</sup>×<sup>300</sup>. This is a layer fully connected to the input where input images are of size 28 × 28 = 784. In order to understand which connections are retained, we can visualize the binary connectivity mask for this layer c<sup>l</sup>=1, by averaging across columns and then reshaping the vector into 2D matrix (*i.e*., c<sup>l</sup>=1 ∈ {0, 1} <sup>784</sup>×<sup>300</sup> → R <sup>784</sup> → R <sup>28</sup>×28). Recall that our method computes c using a minibatch of examples. In this experiment, we curate the mini-batch of examples of the same class and see which weights are retained for that mini-batch of data. We repeat this experiment for all classes

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

Figure 2: Visualizations of pruned parameters of the first layer in LeNet-300-100; the parameters are reshaped to be visualized as an image. Each column represents the visualizations for a particular class obtained using a batch of 100 examples with varying levels of sparsity κ¯, from 10 (top) to 90 (bottom). Bright pixels indicate that the parameters connected to these region had high importance scores (s) and survived from pruning. As the sparsity increases, the parameters connected to the discriminative part of the image for classification survive and the irrelevant parts get pruned.

(*i.e*., digits for MNIST and fashion items for Fashion-MNIST) with varying sparsity levels κ¯. The results are displayed in Figure [2](#page-8-0) (see Appendix [A](#page-12-0) for more results).

The results are significant; important connections seem to reconstruct either the complete image (MNIST) or silhouettes (Fashion-MNIST) of input class. When we use a batch of examples of the digit 0 (*i.e*., the first column of MNIST results), for example, the parameters connected to the foreground of the digit 0 survive from pruning while the majority of background is removed. Also, one can easily determine the identity of items from Fashion-MNIST results. This clearly indicates that our method indeed prunes the *unimportant* connections in performing the classification task, receiving signals only from the most discriminative part of the input. This stands in stark contrast to other pruning methods from which carrying out such inspection is not straightforward.

### 5.5 EFFECTS OF DATA AND WEIGHT INITIALIZATION

Recall that our connection saliency measure depends on the network weights w as well as the given data D (Section [4.2\)](#page-4-0). We study the effect of each of these in this section.

Effect of data. Our connection saliency measure depends on a mini-batch of train examples D<sup>b</sup> (see Algorithm [1\)](#page-4-1). To study the effect of data, we vary the batch size used to compute the saliency (|D<sup>b</sup> |) and check which connections are being pruned as well as how much performance change this results in on the corresponding sparse network. We test with LeNet-300-100 to visualize the remaining parameters, and set the sparsity level κ¯ = 90. Note that the batch size used for training remains the same as 100 for all cases. The results are displayed in Figure [3.](#page-9-0)

Effect of initialization. Our approach prunes a network at a stochastic initialization as discussed. We study the effect of the following initialization methods: 1) RN (random normal), 2) TN (truncated random normal), 3) VS-X (a variance scaling method using [Glorot & Bengio](#page-10-15) [\(2010\)](#page-10-15)), and 4) VS-H (a variance scaling method [He et al. \(2015\)](#page-10-18)). We test on LeNets and RNNs on MNIST and run 20 sets of experiments by varying the seed for initialization. We set the sparsity level κ¯ = 90, and train with Adam optimizer [\(Kingma & Ba \(2015](#page-10-19))) with learning rate of 0.001 without weight decay. Note that for training VS-X initialization is used in all the cases. The results are reported in Figure [3.](#page-9-1)

For all models, VS-H achieves the best performance. The differences between initializers are marginal on LeNets, however, variance scaling methods indeed turns out to be essential for complex RNN models. This effect is significant especially for GRU where without variance scaling initialization, the pruned networks are unable to achieve good accuracies, even with different optimizers. Overall, initializing with a variance scaling method seems crucial to making our saliency measure reliable and model-agnostic.

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

<span id="page-9-1"></span>Figure 3: The effect of different batch sizes: (top-row) survived parameters in the first layer of LeNet-300-100 from pruning visualized as images; (bottom-row) the performance in errors of the pruned networks. For |D<sup>b</sup> | = 1, the sampled example was 8; our pruning precisely retains the valid connections. As |D<sup>b</sup> | increases, survived parameters get close to the average of all examples in the train set (last column), and the error decreases.

| Init. | LeNet-300-100 | LeNet-5-Caffe | LSTM-s        | GRU-s           |
|-------|---------------|---------------|---------------|-----------------|
| RN    | 1.90 ± (0.09) | 0.89 ± (0.04) | 2.93 ± (0.20) | 47.61 ± (20.49) |
| TN    | 1.96 ± (0.11) | 0.87 ± (0.05) | 3.03 ± (0.17) | 46.48 ± (22.25) |
| VS-X  | 1.91 ± (0.10) | 0.88 ± (0.07) | 1.48 ± (0.09) | 1.80 ± (0.10)   |
| VS-H  | 1.88 ± (0.10) | 0.85 ± (0.05) | 1.47 ± (0.08) | 1.80 ± (0.14)   |

Table 3: The effect of initialization on our saliency score. We report the classification errors (±std). Variance scaling initialization (VS-X, VS-H) improves the performance, especially for RNNs.

#### 5.6 FITTING RANDOM LABELS

To further explore the use cases of SNIP, we run the experiment introduced in [Zhang et al.](#page-11-23) [\(2017\)](#page-11-23) and check whether the sparse network obtained by SNIP memorizes the dataset. Specifically, we train LeNet-5-Caffe for both the reference model and pruned model (with κ¯ = 99) on MNIST with either true or randomly shuffled labels. To compute the connection sensitivity, always true labels are used. The results are plotted in Figure [4.](#page-9-2)

Given true labels, both the reference (red) and pruned (blue) models quickly reach to almost zero training loss. However, the reference model provided with random labels (green) also reaches to very low training loss, even

<span id="page-9-2"></span>![](_page_9_Figure_8.jpeg)

Figure 4: The sparse model pruned by SNIP does not fit the random labels.

with an explicit L2 regularizer (purple), indicating that neural networks have enough capacity to memorize completely random data. In contrast, the model pruned by SNIP (orange) fails to fit the random labels (high training error). The potential explanation is that the pruned network does not have sufficient capacity to fit the random labels, but it is able to classify MNIST with true labels, reinforcing the significance of our saliency criterion. It is possible that a similar experiment can be done with other pruning methods [\(Molchanov et al.](#page-11-6) [\(2017a](#page-11-6))), however, being simple, SNIP enables such exploration much easier. We provide a further analysis on the effect of varying κ¯ in Appendix [B.](#page-13-0)

### 6 DISCUSSION AND FUTURE WORK

In this work, we have presented a new approach, SNIP, that is simple, versatile and interpretable; it prunes irrelevant connections for a given task at single-shot prior to training and is applicable to a variety of neural network models without modifications. While SNIP results in extremely sparse models, we find that our connection sensitivity measure itself is noteworthy in that it diagnoses important connections in the network from a purely untrained network. We believe that this opens up new possibilities beyond pruning in the topics of understanding of neural network architectures, multi-task transfer learning and structural regularization, to name a few. In addition to these potential directions, we intend to explore the generalization capabilities of sparse networks.

#### ACKNOWLEDGEMENTS

This work was supported by the Korean Government Graduate Scholarship, the ERC grant ERC-2012-AdG 321162-HELIOS, EPSRC grant Seebibyte EP/M013774/1 and EPSRC/MURI grant EP/N019474/1. We would also like to acknowledge the Royal Academy of Engineering and FiveAI.

### REFERENCES

- <span id="page-10-0"></span>Sanjeev Arora, Rong Ge, Behnam Neyshabur, and Yi Zhang. Stronger generalization bounds for deep nets via a compression approach. *ICML*, 2018.
- <span id="page-10-13"></span>Leo Breiman. Better subset regression using the nonnegative garrote. *Technometrics*, 1995.
- <span id="page-10-3"></span>Miguel A. Carreira-Perpi ˜n´an and Yerlan Idelbayev. "Learning-c ´ ompression" algorithms for neural net pruning. *CVPR*, 2018.
- <span id="page-10-2"></span>Yves Chauvin. A back-propagation algorithm with optimal use of hidden units. *NIPS*, 1989.
- <span id="page-10-17"></span>Kyunghyun Cho, Bart Van Merri¨enboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. *EMNLP*, 2014.
- <span id="page-10-16"></span>Xin Dong, Shangyu Chen, and Sinno Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. *NIPS*, 2017.
- <span id="page-10-15"></span>Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 2010.
- <span id="page-10-9"></span>Yunchao Gong, Liu Liu, Ming Yang, and Lubomir Bourdev. Compressing deep convolutional networks using vector quantization. *arXiv preprint arXiv:1412.6115*, 2014.
- <span id="page-10-14"></span>Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. Deep learning. *MIT press Cambridge*, 2016.
- <span id="page-10-7"></span>Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. *NIPS*, 2016.
- <span id="page-10-10"></span>Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, and Pritish Narayanan. Deep learning with limited numerical precision. *ICML*, 2015.
- <span id="page-10-1"></span>Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. *NIPS*, 2015.
- <span id="page-10-6"></span>Babak Hassibi, David G Stork, and Gregory J Wolff. Optimal brain surgeon and general network pruning. *Neural Networks*, 1993.
- <span id="page-10-18"></span>Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. *ICCV*, 2015.
- <span id="page-10-11"></span>Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. *NIPS*, 2016.
- <span id="page-10-4"></span>Masumi Ishikawa. Structural learning with forgetting. *Neural Networks*, 1996.
- <span id="page-10-8"></span>Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Speeding up convolutional neural networks with low rank expansions. *BMVC*, 2014.
- <span id="page-10-5"></span>Ehud D Karnin. A simple procedure for pruning back-propagation trained neural networks. *Neural Networks*, 1990.
- <span id="page-10-19"></span>Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *ICLR*, 2015.
- <span id="page-10-12"></span>Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. *ICML*, 2017.
- <span id="page-11-16"></span>Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. 2012.
- <span id="page-11-21"></span>Quoc V Le, Navdeep Jaitly, and Geoffrey E Hinton. A simple way to initialize recurrent networks of rectified linear units. *CoRR*, 2015.
- <span id="page-11-1"></span>Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. *NIPS*, 1990.
- <span id="page-11-14"></span>Yann LeCun, L´eon Bottou, Genevieve B. Orr, and Klaus-Robert M ¨uller. Efficient backprop. *Neural Networks: Tricks of the Trade*, 1998.
- <span id="page-11-8"></span>Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. *ICLR*, 2017.
- <span id="page-11-15"></span>Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l<sup>0</sup> regularization. *ICLR*, 2018.
- <span id="page-11-13"></span>Decebal Constantin Mocanu, Elena Mocanu, Peter Stone, Phuong H Nguyen, Madeleine Gibescu, and Antonio Liotta. Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science. *Nature Communications*, 2018.
- <span id="page-11-6"></span>Dmitry Molchanov, Arsenii Ashukha, and Dmitry Vetrov. Variational dropout sparsifies deep neural networks. *ICML*, 2017a.
- <span id="page-11-9"></span>Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. *ICLR*, 2017b.
- <span id="page-11-0"></span>Michael C Mozer and Paul Smolensky. Skeletonization: A technique for trimming the fat from a network via relevance assessment. *NIPS*, 1989.
- <span id="page-11-4"></span>Sharan Narang, Erich Elsen, Gregory Diamos, and Shubho Sengupta. Exploring sparsity in recurrent neural networks. *ICLR*, 2017.
- <span id="page-11-11"></span>Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. *NIPS*, 2015.
- <span id="page-11-7"></span>Steven J Nowlan and Geoffrey E Hinton. Simplifying neural networks by soft weight-sharing. *Neural Computation*, 1992.
- <span id="page-11-12"></span>Ameya Prabhu, Girish Varma, and Anoop Namboodiri. Deep expander networks: Efficient deep networks from graph theory. *ECCV*, 2018.
- <span id="page-11-2"></span>Russell Reed. Pruning algorithms-a survey. *Neural Networks*, 1993.
- <span id="page-11-22"></span>Abigail See, Minh-Thang Luong, and Christopher D Manning. Compression of neural machine translation models via pruning. *CoNLL*, 2016.
- <span id="page-11-17"></span>Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *ICLR*, 2015.
- <span id="page-11-5"></span>Karen Ullrich, Edward Meeds, and Max Welling. Soft weight-sharing for neural network compression. *ICLR*, 2017.
- <span id="page-11-3"></span>Andreas S Weigend, David E Rumelhart, and Bernardo A Huberman. Generalization by weightelimination with application to forecasting. *NIPS*, 1991.
- <span id="page-11-10"></span>Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. *NIPS*, 2016.
- <span id="page-11-18"></span>Sergey Zagoruyko. 92.45% on cifar-10 in torch. *Torch Blog*, 2015.
- <span id="page-11-19"></span>Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. *BMVC*, 2016.
- <span id="page-11-20"></span>Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. *arXiv preprint arXiv:1409.2329*, 2014.
- <span id="page-11-23"></span>Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. *ICLR*, 2017.

### <span id="page-12-1"></span><span id="page-12-0"></span>A VISUALIZING PRUNED PARAMETERS ON (INVERTED) (FASHION-)MNIST

![](_page_12_Figure_2.jpeg)

Figure 5: Results of pruning with SNIP on inverted (Fashion-)MNIST (*i.e*., dark and bright regions are swapped). Notably, even if the data is inverted, the results are the same as the ones on the original (Fashion-)MNIST in Figure [2.](#page-8-0)

![](_page_12_Figure_4.jpeg)

Figure 6: Results of pruning with ∂L/∂w on the original and inverted (Fashion-)MNIST. Notably, compared to the case of using SNIP (Figures [2](#page-8-0) and [5\)](#page-12-1), the results are different: Firstly, the results on the original (Fashion-)MNIST (*i.e*., (a) and (c) above) are not the same as the ones using SNIP (*i.e*., (a) and (b) in Figure [2\)](#page-8-0). Moreover, the pruning patterns are inconsistent with different sparsity levels, either intra-class or inter-class. Furthermore, using ∂L/∂w results in different pruning patterns between the original and inverted data in some cases (*e.g*., the 2 nd columns between (c) and (d)).

![](_page_13_Figure_1.jpeg)

### <span id="page-13-0"></span>B FITTING RANDOM LABELS: VARYING SPARSITY LEVELS

Figure 7: The effect of varying sparsity levels (κ¯). The lower κ¯ becomes, the lower training loss is recorded, meaning that a network with more parameters is more vulnerable to fitting random labels. Recall, however, that all pruned models are able to learn to perform the classification task without losing much accuracy (see Figure [1\)](#page-5-0). This potentially indicates that the pruned network does not have sufficient capacity to fit the random labels, but it is capable of performing the classification.

## C TINY-IMAGENET

| Architecture  | Model                                                | Sparsity (%)                         | # Parameters                                                                              | Error (%)                                                                                        | ∆                                         |
|---------------|------------------------------------------------------|--------------------------------------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------|
| Convolutional | AlexNet-s<br>AlexNet-b<br>VGG-C<br>VGG-D<br>VGG-like | 90.0<br>90.0<br>95.0<br>95.0<br>95.0 | 5.1m →<br>507k<br>8.5m →<br>849k<br>10.5m →<br>526k<br>15.2m →<br>762k<br>15.0m →<br>749k | 62.52 →<br>65.27<br>62.76 →<br>65.54<br>56.49 →<br>57.48<br>56.85 →<br>57.00<br>54.86 →<br>55.73 | +2.75<br>+2.78<br>+0.99<br>+0.15<br>+0.87 |

Table 4: Pruning results of SNIP on Tiny-ImageNet (before → after). Tiny-ImageNet[2](#page-13-1) is a subset of the full ImageNet: there are 200 classes in total, each class has 500 and 50 images for training and validation respectively, and each image has the spatial resolution of 64×64. Compared to CIFAR-10, the resolution is doubled, and to deal with this, the stride of the first convolution in all architectures is doubled, following the standard practice for this dataset. In general, the Tiny-ImageNet classification task is considered much more complex than MNIST or CIFAR-10. Even on Tiny-ImageNet, however, SNIP is still able to prune a large amount of parameters with minimal loss in performance. AlexNet models lose more accuracies than VGGs, which may be attributed to the fact that the first convolution stride for AlexNet is set to be 4 (by its design of no pooling) which is too large and could lead to high loss of information when pruned.

<span id="page-13-1"></span><sup>2</sup> https://tiny-imagenet.herokuapp.com/

### D ARCHITECTURE DETAILS

| Module | Weight                     | Stride | Bias          | BatchNorm | ReLU |
|--------|----------------------------|--------|---------------|-----------|------|
| Conv   | [11, 11, 3, 96]            | [2, 2] | [96]          | X         | X    |
| Conv   | [5, 5, 96, 256]            | [2, 2] | [256]         | X         | X    |
| Conv   | [3, 3, 256, 384]           | [2, 2] | [384]         | X         | X    |
| Conv   | [3, 3, 384, 384]           | [2, 2] | [384]         | X         | X    |
| Conv   | [3, 3, 384, 256]           | [2, 2] | [256]         | X         | X    |
| Linear | [256, 1024 ×<br>k]         | –      | [1024 ×<br>k] | X         | X    |
| Linear | [1024 ×<br>k, 1024 ×<br>k] | –      | [1024 ×<br>k] | X         | X    |
| Linear | [1024 ×<br>k, c]           | –      | [c]           | ✗         | ✗    |

Table 5: AlexNet-s (k = 1) and AlexNet-b (k = 2). In the last layer, c denotes the number of possible classes: c = 10 for CIFAR-10 and c = 200 for Tiny-ImageNet. The strides in the first convolution layer for Tiny-ImageNet are set [4, 4] instead of [2, 2] to deal with the increase in the image resolution.

| Module | Weight                   | Stride | Bias  | BatchNorm | ReLU |
|--------|--------------------------|--------|-------|-----------|------|
| Conv   | [3, 3, 3, 64]            | [1, 1] | [64]  | X         | X    |
| Conv   | [3, 3, 64, 64]           | [1, 1] | [64]  | X         | X    |
| Pool   | –                        | [2, 2] | –     | ✗         | ✗    |
| Conv   | [3, 3, 64, 128]          | [1, 1] | [128] | X         | X    |
| Conv   | [3, 3, 128, 128]         | [1, 1] | [128] | X         | X    |
| Pool   | –                        | [2, 2] | –     | ✗         | ✗    |
| Conv   | [3, 3, 128, 256]         | [1, 1] | [256] | X         | X    |
| Conv   | [3, 3, 256, 256]         | [1, 1] | [256] | X         | X    |
| Conv   | [1/3/3, 1/3/3, 256, 256] | [1, 1] | [256] | X         | X    |
| Pool   | –                        | [2, 2] | –     | ✗         | ✗    |
| Conv   | [3, 3, 256, 512]         | [1, 1] | [512] | X         | X    |
| Conv   | [3, 3, 512, 512]         | [1, 1] | [512] | X         | X    |
| Conv   | [1/3/3, 1/3/3, 512, 512] | [1, 1] | [512] | X         | X    |
| Pool   | –                        | [2, 2] | –     | ✗         | ✗    |
| Conv   | [3, 3, 512, 512]         | [1, 1] | [512] | X         | X    |
| Conv   | [3, 3, 512, 512]         | [1, 1] | [512] | X         | X    |
| Conv   | [1/3/3, 1/3/3, 512, 512] | [1, 1] | [512] | X         | X    |
| Pool   | –                        | [2, 2] | –     | ✗         | ✗    |
| Linear | [512, 512]               | –      | [512] | X         | X    |
| Linear | [512, 512]               | –      | [512] | X         | X    |
| Linear | [512, c]                 | –      | [c]   | ✗         | ✗    |

Table 6: VGG-C/D/like. In the last layer, c denotes the number of possible classes: c = 10 for CIFAR-10 and c = 200 for Tiny-ImageNet. The strides in the first convolution layer for Tiny-ImageNet are set [2, 2] instead of [1, 1] to deal with the increase in the image resolution. The second Linear layer is only used in VGG-C/D.