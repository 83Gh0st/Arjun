# ML Fundamentals Interview Master Guide (2026)
### Complete answer bank — classic ML, deep learning, statistical ML, LLMs/GenAI, and multimodal — written so you can speak each answer out loud and survive cross-questioning.

**How to use this doc:** Every topic is written as (1) a plain-English explanation assuming zero prior context, (2) the mechanism/math/architecture, (3) why it matters / when to use it, and (4) a "likely cross-questions" block so you're not caught off guard. Read top to bottom once, then use the cross-question blocks as a self-test.

---

## Table of Contents

**Part I — Classic ML Foundations**
1. [Types of ML & Algorithm Categories](#p1-1)
2. [Linear Models: Linear & Logistic Regression](#p1-2)
3. [SVM & Linear Discriminant Analysis](#p1-3)
4. [Decision Trees](#p1-4)
5. [Ensemble Methods (Bagging, Boosting, RF, XGBoost)](#p1-5)
6. [Optimization: Gradient Descent & Variants](#p1-6)
7. [Loss Functions](#p1-7)
8. [Feature Selection & Importance](#p1-8)
9. [Model Evaluation & Metrics](#p1-9)
10. [Cross-Validation & Model Selection](#p1-10)
11. [Unsupervised Learning: Clustering](#p1-11)
12. [GMM, LSA, HMM](#p1-12)
13. [Dimensionality Reduction (PCA, ICA, t-SNE)](#p1-13)
14. [Bias-Variance Tradeoff & Regularization](#p1-14)
15. [Sampling Techniques](#p1-15)
16. [Handling Messy Data (Missing, Imbalanced, Drift)](#p1-16)

**Part II — Deep Learning**
17. [Feedforward Neural Networks & Backpropagation](#p2-1)
18. [RNNs, BPTT, Vanishing/Exploding Gradients](#p2-2)
19. [LSTM & Dropout in Sequence Models](#p2-3)
20. [Seq2Seq & Attention](#p2-4)
21. [Transformer Architecture — Full Deep Dive](#p2-5)
22. [Embeddings](#p2-6)

**Part III — Statistical ML & Misc**
23. [Bayesian Methods: Naive Bayes, MAP, MLE](#p3-1)
24. [Statistical Significance: R², p-values](#p3-2)
25. [Outliers & Similarity Metrics](#p3-3)

**Part IV — LLMs & Foundation Models (2026)**
26. [Attention at Scale: MHA → MQA → GQA → MLA](#p4-1)
27. [Positional Encodings: RoPE, ALiBi, Long Context](#p4-2)
28. [KV Cache & FlashAttention](#p4-3)
29. [Modern Block Internals: RMSNorm, SwiGLU, Pre-Norm](#p4-4)
30. [Mixture-of-Experts (MoE)](#p4-5)
31. [Tokenization & Scaling Laws](#p4-6)
32. [Training Pipeline Overview](#p4-7)
33. [Post-Training Zoo: SFT, RLHF/PPO, DPO, SimPO, KTO, ORPO, GRPO, RLVR](#p4-8)
34. [PEFT: LoRA & QLoRA](#p4-9)
35. [Inference & Serving: Quantization, Paged Attention, Speculative Decoding](#p4-10)
36. [Decoding Strategies & In-Context Learning](#p4-11)
37. [Evaluating LLMs & RAG Systems](#p4-12)

**Part V — Multimodal & Generative AI**
38. [Multimodal Foundation Models](#p5-1)
39. [Vision-Language Models (VLMs)](#p5-2)
40. [Vision-Language-Action Models (VLAs)](#p5-3)
41. [Diffusion vs Autoregressive Generation](#p5-4)

**Part VI — Rapid-Fire Cheat Sheet**
**Part VII — Further Resources**

---

## Part I — Classic ML Foundations

<a name="p1-1"></a>
### 1. Types of ML & Algorithm Categories

**What is machine learning, and how is it different from traditional programming?**
In traditional programming you write explicit rules (`if/else` logic) that transform input → output. In machine learning, you give the algorithm input **and** the desired output (or just input, or partial output), and it *learns the rules* — a function `f(x) ≈ y` — by optimizing parameters against data, rather than a human hand-coding them. The core idea: instead of programming behavior, you program a **learning procedure** that discovers behavior from data.

**The three (four) main categories:**

- **Supervised learning** — you have labeled data `(x, y)` pairs. The model learns a mapping from features to a known target.
  - *Classification*: target is categorical (spam/not spam, digit 0-9).
  - *Regression*: target is continuous (house price, temperature).
  - Examples: linear regression, logistic regression, decision trees, SVMs, neural nets.
- **Unsupervised learning** — you only have `x`, no labels. The model finds structure — groups, densities, lower-dimensional representations — on its own.
  - *Clustering* (k-means, hierarchical, DBSCAN), *dimensionality reduction* (PCA, t-SNE), *density estimation* (GMM).
- **Semi-supervised learning** — a small amount of labeled data + a large amount of unlabeled data. Common when labeling is expensive (e.g., medical imaging). Techniques: self-training (pseudo-labeling), label propagation on a similarity graph, consistency regularization.
- **Reinforcement learning** (often mentioned as a 4th, though not in classic taxonomy above) — an agent learns by interacting with an environment, receiving rewards/penalties, and optimizing a policy. Relevant today because RLHF/RLVR (post-training for LLMs) is RL applied to language generation.

**Parametric vs. non-parametric algorithms**
- *Parametric*: the model has a **fixed number of parameters** regardless of dataset size (e.g., linear regression has `d+1` weights no matter if you have 100 or 100M rows). Assumes a specific functional form. Fast, low variance, but can underfit if the true function doesn't match the assumed form (high bias risk).
- *Non-parametric*: the **effective number of parameters grows with the data** (e.g., k-NN, kernel SVM, decision trees). More flexible, can model complex functions, but needs more data and is prone to overfitting/high variance, and is often slower at inference.
- *Cross-question: "Is a neural network parametric?"* — Yes, technically parametric (fixed architecture = fixed parameter count once you pick it), but in practice treated as "practically non-parametric" because you can scale parameter count arbitrarily with depth/width to fit almost any data distribution.

**Linear vs. nonlinear algorithms**
- *Linear*: decision boundary or regression surface is a linear combination of inputs, `w·x + b` (linear regression, logistic regression, linear SVM, LDA). Fast, interpretable, high bias if true relationship is nonlinear.
- *Nonlinear*: can model curved decision boundaries/relationships (kernel SVM, decision trees, neural nets, k-NN). More flexible, more prone to overfitting.

---

<a name="p1-2"></a>
### 2. Linear Models: Linear & Logistic Regression

#### Linear Regression
**Goal:** predict a continuous value `y` from features `x` using `ŷ = w·x + b`.

**Least squares:** we choose `w, b` to minimize the **sum of squared residuals**:
```
J(w, b) = (1/n) * Σ (y_i - ŷ_i)²         where ŷ_i = w·x_i + b
```
A **residual** is `y_i - ŷ_i` — the vertical distance between the actual point and the fitted line. We square residuals so (a) positive and negative errors don't cancel, and (b) large errors are penalized disproportionately (this also makes the loss differentiable and convex).

**Closed-form solution (Normal Equation):** `w = (XᵀX)⁻¹ Xᵀy`. This exists because squared-error loss is convex and quadratic in `w` — you can solve for the minimum analytically. In practice, gradient descent is used when `X` is huge (matrix inversion is O(d³), expensive for high-dimensional data) or when `XᵀX` isn't invertible (multicollinearity) — regularization (Ridge) fixes that.

**Multivariate regression** simply means more than one feature (`x` is a vector, not a scalar) — don't confuse with "multiple regression" (many features) vs. "multivariate regression" (multiple *target* variables predicted jointly); interviewers sometimes test whether you know this distinction.

**Assumptions of linear regression** (classic cross-question): linearity between X and y, independence of errors, homoscedasticity (constant variance of residuals), normality of residuals (for inference/p-values, not prediction), no severe multicollinearity.

#### Logistic Regression
**Goal:** binary classification. Despite the name, it's a classifier, not a regressor.

**Mechanism:** compute a linear score `z = w·x + b`, then squash it into a probability with the **sigmoid function**:
```
σ(z) = 1 / (1 + e^(-z))          →  outputs value in (0, 1)
ŷ = P(y=1 | x) = σ(w·x + b)
```
Sigmoid is chosen because it maps `(-∞, ∞) → (0, 1)`, is smooth/differentiable (needed for gradient descent), and its output can be interpreted as a probability. Decision boundary: predict class 1 if `σ(z) ≥ 0.5`, i.e., `z ≥ 0`, i.e., logistic regression's decision boundary is **linear in x**.

**Cost function — Binary Cross-Entropy (log loss):**
```
J(w) = -(1/n) * Σ [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]
```
Why not squared error like linear regression? Because squared error with a sigmoid is **non-convex** (multiple local minima), making gradient descent unreliable. Cross-entropy, derived from maximum likelihood estimation of a Bernoulli distribution, IS convex in `w` for logistic regression, giving a single global minimum.

**Gradient (used for gradient descent), and why it's elegant:**
```
∂J/∂w = (1/n) * Σ (ŷ_i - y_i) * x_i
```
This has the *exact same form* as linear regression's gradient — that's a well-known interview "gotcha" fact: even though the cost functions differ, the gradient of cross-entropy w.r.t. weights simplifies to `(prediction - actual) * feature`, because the sigmoid derivative and the `1/ŷ(1-ŷ)` term from the cross-entropy derivative cancel out perfectly.

**Minimal code sketch (for a "code it" round):**
```python
def sigmoid(z): return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, lr=0.1, epochs=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0
    for _ in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        dw = (1/n) * X.T @ (y_hat - y)
        db = (1/n) * np.sum(y_hat - y)
        w -= lr * dw
        b -= lr * db
    return w, b
```

**Cross-questions:**
- *"Why is logistic regression called regression if it classifies?"* — Historically it regresses the **log-odds** (logit): `log(p/(1-p)) = w·x + b` is a linear regression on the logit-transformed probability.
- *"How do you extend logistic regression to multi-class?"* — Softmax regression (multinomial logistic regression) generalizes sigmoid to multiple classes: `P(y=k|x) = e^{z_k} / Σ_j e^{z_j}`, paired with categorical cross-entropy.
- *"What if classes are not linearly separable?"* — Add polynomial/interaction features, or use a kernel method / nonlinear model instead.

---

<a name="p1-3"></a>
### 3. SVM & Linear Discriminant Analysis

#### Support Vector Machines (SVM)
**Core idea:** find the hyperplane that separates two classes with the **maximum margin** — the largest possible distance to the nearest points of either class (the "support vectors"). Intuition: a wider margin generalizes better to unseen data than any separating boundary.

**Formulation:** minimize `(1/2)||w||²` subject to `y_i(w·x_i + b) ≥ 1` for all `i` (hard margin — requires perfectly separable data). For real-world (non-separable) data, we use the **soft margin** with slack variables `ξ_i`:
```
min (1/2)||w||² + C * Σ ξ_i         s.t.  y_i(w·x_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```
`C` is the regularization strength: **large C** = less tolerance for misclassification (narrower margin, can overfit); **small C** = wider margin, more tolerant of misclassified points (can underfit). This maps to the **hinge loss** formulation: `L = Σ max(0, 1 - y_i(w·x_i+b)) + λ||w||²`.

**Kernel trick / Kernel SVM:** when data isn't linearly separable in the original space, we implicitly map `x → φ(x)` into a higher-dimensional space where it might be separable, **without ever computing `φ(x)` explicitly** — we just need a kernel function `K(x_i, x_j) = φ(x_i)·φ(x_j)` that computes the dot product in that space directly. Common kernels:
- Linear: `K(x,z) = x·z`
- Polynomial: `K(x,z) = (x·z + c)^d`
- **RBF/Gaussian** (most common): `K(x,z) = exp(-γ||x-z||²)` — maps to infinite-dimensional space; `γ` controls how far the influence of a single point reaches (high γ = tight, wiggly boundary → overfit risk; low γ = smooth boundary → underfit risk).

**Why does this work?** Because SVM's dual optimization problem only ever needs **dot products** between data points, not the points themselves — so swapping in a kernel function lets you work in a transformed space "for free," computationally.

**Cross-questions:**
- *"SVM vs logistic regression?"* — SVM maximizes margin (geometric), logistic regression maximizes likelihood (probabilistic); logistic regression naturally outputs calibrated probabilities, SVM doesn't (needs Platt scaling); SVM with hinge loss is less sensitive to outliers far from the margin (loss=0 once correctly classified beyond margin) while logistic regression's loss is never zero.
- *"Why maximize margin?"* — Statistical learning theory (VC dimension) shows wider margins → better generalization bounds.

#### Linear Discriminant Analysis (LDA)
**Core idea:** LDA is both a classifier and a dimensionality-reduction technique. It finds the linear projection of the data that **maximizes the ratio of between-class variance to within-class variance** — i.e., it finds an axis where classes are pushed apart from each other but each class is tight internally.

**Assumptions:** each class is Gaussian-distributed, and (critically) **all classes share the same covariance matrix**. Under this assumption, the optimal (Bayes) decision boundary between classes is provably linear — that's where "linear" in LDA comes from.

**LDA vs PCA (classic cross-question):** PCA is unsupervised — it finds directions of maximum *variance* in the data, ignoring labels. LDA is supervised — it finds directions that maximize *class separability*. PCA can throw away information that's useless for reconstruction but crucial for classification; LDA is built for classification.

**LDA vs Logistic Regression:** LDA assumes Gaussian class-conditional densities with shared covariance (generative model — models `P(x|y)` then applies Bayes' rule); logistic regression directly models `P(y|x)` (discriminative). If the Gaussian assumption holds, LDA is more statistically efficient (needs less data); if it's violated, logistic regression is more robust because it makes no distributional assumption on `x`.

---

<a name="p1-4"></a>
### 4. Decision Trees

**Core idea:** recursively split the feature space into regions using simple threshold rules (`feature ≤ value?`) forming a tree; each leaf makes a prediction (majority class for classification, mean value for regression).

**Structure:**
- **Root/internal nodes**: hold a split condition on one feature.
- **Branches**: outcome of the condition (yes/no).
- **Leaves**: hold the final prediction (a class label or a numeric value — in regression trees, the mean of training targets in that region).

**Training algorithm (greedy, top-down, recursive partitioning — "CART"-style):**
1. At the current node, consider every feature and every possible split threshold.
2. For each candidate split, compute an **impurity** measure for the two resulting child nodes.
3. Pick the split that gives the largest **impurity reduction** (information gain).
4. Recurse on each child until a stopping criterion is met.

**Impurity / split criteria:**
- **Gini impurity** (classification): `Gini = 1 - Σ p_k²` — probability that a randomly picked sample would be mislabeled if labeled randomly according to the class distribution in that node. Ranges 0 (pure) to 0.5 (max impurity, binary case).
- **Entropy** (classification): `H = -Σ p_k log₂(p_k)`. **Information Gain** = entropy(parent) − weighted average entropy(children). Entropy is more computationally expensive (log) but sometimes gives slightly different splits than Gini; in practice they rarely differ much.
- **Variance reduction / MSE** (regression trees): minimize the weighted variance of targets in child nodes.

**"Logits" in trees** — this usually refers to **leaf values in gradient-boosted trees** (e.g., XGBoost) which store a raw score (like a logit) that gets passed through a sigmoid/softmax at the end, rather than a direct class probability, since boosting adds scores across many trees.

**Stopping criteria (stop growing the tree when):** max depth reached, minimum samples per leaf/split not met, node is pure (impurity = 0), impurity reduction from best split falls below a threshold, or max number of leaf nodes reached.

**Inference:** walk from the root, following the branch matching the sample's feature value at each node, until you hit a leaf; return the leaf's stored value/class.

**Pruning:** unconstrained trees overfit badly (they can grow until every leaf has 1 sample = 100% train accuracy, 0% generalization). Two approaches:
- **Pre-pruning** (early stopping): limit `max_depth`, `min_samples_leaf`, `min_impurity_decrease` during growth.
- **Post-pruning** (e.g., **cost-complexity pruning** used in CART): grow the full tree, then trim back branches that don't improve validation performance enough, using a penalty `α` on tree size: `Cost = Error + α * (#leaves)`. Increase `α` to prune more aggressively; choose `α` via cross-validation.

**Pros:** interpretable (can visualize/explain), no feature scaling needed, handles nonlinear relationships and feature interactions naturally, handles mixed data types.
**Cons:** high variance (small data changes → very different tree — this is *why* ensembles like Random Forest exist), prone to overfitting if unpruned, axis-aligned splits only (struggles with diagonal decision boundaries), greedy algorithm isn't globally optimal.

**Cross-questions:**
- *"Why greedy and not globally optimal?"* — Finding the globally optimal tree is NP-hard; greedy top-down splitting is a tractable heuristic approximation.
- *"Gini vs Entropy — which to use?"* — Gini is slightly faster (no log), tends to isolate the most frequent class in its own branch; entropy is more sensitive to changes in class probabilities. In practice, negligible difference in output quality; Gini is the default in scikit-learn/CART.

---

<a name="p1-5"></a>
### 5. Ensemble Methods (Bagging, Boosting, Random Forest, XGBoost)

**Why ensembles?** Combine many "weak" models into a "strong" model. Two big families, based on **how they reduce error**:
- **Bagging** reduces **variance** (by averaging independent, high-variance models).
- **Boosting** reduces **bias** (by sequentially correcting the errors of prior weak models).

#### Bagging (Bootstrap Aggregating)
1. Create `B` bootstrap samples (sample training data **with replacement**, same size as original — each bootstrap sample contains ~63.2% unique original rows).
2. Train one model (typically a high-variance model like a deep decision tree) independently on each bootstrap sample.
3. Aggregate predictions: majority vote (classification) or average (regression).

Because each model is trained on a different random subset and errors are (roughly) independent, averaging **cancels out variance** without increasing bias much (assuming trees are unbiased/deep). This is why bagging works best with **high-variance, low-bias base learners** like unpruned decision trees.

**Random Forest** = bagging of decision trees **+ one more trick**: at each split, only consider a **random subset of features** (typically `√d` for classification, `d/3` for regression) instead of all features. This **decorrelates the trees** — without it, if one feature is very predictive, every bootstrapped tree would pick it at the top split, making trees highly correlated and reducing the variance-cancelling benefit of averaging. Random feature subsampling forces trees to be more diverse.
- Also gives a free validation estimate: **Out-of-Bag (OOB) error** — since each tree only sees ~63% of data, the remaining ~37% ("out-of-bag" samples) can be used to validate that tree without a separate holdout set.
- **Feature importance** from RF: computed via mean decrease in impurity (how much each feature reduces Gini/entropy on average across all trees and splits) or permutation importance (shuffle a feature's values and see how much performance drops).

#### Boosting
Train models **sequentially**, where each new model focuses on correcting the mistakes of the ensemble so far. Unlike bagging, boosting reduces **bias** (turns weak learners into a strong learner) but can increase variance if run too long (risk of overfitting), which is why boosting needs regularization (learning rate, tree depth limits, early stopping).

**AdaBoost (Adaptive Boosting):**
1. Initialize equal weights on all training samples.
2. Train a weak learner (often a shallow "decision stump" — 1-level tree).
3. Compute the weighted error rate; compute the learner's "say" (weight in final vote) `α = 0.5 * ln((1-err)/err)` — more accurate learners get more say.
4. **Increase weights on misclassified samples**, decrease weights on correctly classified ones, so the next learner focuses more on the hard examples.
5. Repeat; final prediction = weighted vote of all weak learners: `sign(Σ α_t * h_t(x))`.

**Gradient Boosting Machines (GBM):** generalizes AdaBoost — instead of reweighting samples, each new tree is trained to predict the **negative gradient (residual) of the loss function** with respect to the current ensemble's predictions. For squared-error loss, the negative gradient is literally `y - ŷ` (the residual), so GBM intuitively "fits each new tree to the errors of the previous ensemble." A **learning rate (shrinkage)** `η` scales each tree's contribution: `F_t(x) = F_{t-1}(x) + η * h_t(x)` — smaller η needs more trees but generalizes better (classic bias-variance trade with number of estimators).

**XGBoost (Extreme Gradient Boosting):** an engineered, regularized, faster implementation of gradient boosting. Key improvements over vanilla GBM:
- **Regularized objective**: adds `Ω(tree) = γ*T + (1/2)λ*Σw_j²` (T = number of leaves, w_j = leaf weights) directly into the loss, penalizing tree complexity — built-in L1/L2 regularization on leaf weights.
- **Second-order optimization**: uses both gradient AND **Hessian** (second derivative) of the loss to find optimal leaf values (Newton's method style) — more accurate step direction than first-order GBM.
- **Efficient split-finding**: approximate/histogram-based algorithms, handles sparse data (missing values learn a default split direction).
- **System-level speed**: parallelized split-finding (across features, not trees — trees are still sequential), cache-awareness, out-of-core computation for huge datasets.
- Built-in handling for missing values, column subsampling (like RF) for extra regularization/diversity.

**Other important boosting variants (good to name-drop):** LightGBM (leaf-wise growth + histogram binning, very fast on large data), CatBoost (native categorical feature handling via ordered target statistics, reduces prediction shift/leakage).

**Cross-questions:**
- *"Bagging vs Boosting — when to use which?"* — Bagging (RF) when base model overfits (high variance) and you want stability/parallelizable training. Boosting (XGBoost) when base model underfits (high bias, e.g., shallow trees) and you want maximum accuracy, and can afford sequential training + careful tuning to avoid overfitting.
- *"Why do trees work well as base learners for both bagging and boosting?"* — They're flexible, handle nonlinearity/interactions/mixed data, and their bias/variance profile can be tuned via depth to suit either method (deep=low bias/high variance for bagging, shallow=high bias/low variance stumps for boosting).
- *"How does XGBoost handle missing values?"* — Learns a default direction for missing values at each split during training, based on which direction minimizes loss — doesn't require imputation.

---

<a name="p1-6"></a>
### 6. Optimization: Gradient Descent & Variants

**Core idea:** most ML training is "minimize a loss function `J(θ)` over parameters `θ`." Gradient descent does this iteratively by moving parameters in the direction that decreases the loss fastest — the **negative gradient**.

```
θ_{t+1} = θ_t - η * ∇J(θ_t)
```
`η` = learning rate (step size). `∇J(θ_t)` = gradient (vector of partial derivatives) evaluated at current parameters.

**Intuition:** the gradient points in the direction of steepest *ascent*; moving opposite to it locally decreases the loss the most. Repeat until convergence (gradient ≈ 0, or loss stops improving).

**Learning rate trade-off:** too large → overshoots/diverges/oscillates; too small → painfully slow convergence, can get stuck in flat regions. Often scheduled to decay over training (learning-rate schedules: step decay, cosine annealing, warmup + decay — the last is standard in LLM training).

**Batch / Full-batch Gradient Descent:** compute gradient using the **entire training set** per update. Accurate gradient direction, but very slow per step for large datasets, and can get stuck more easily in sharp local minima/saddle points because there's no noise to escape them.

**Stochastic Gradient Descent (SGD):** compute gradient using **just one sample** per update. Extremely noisy but very fast per step, and the noise itself can help escape shallow local minima / saddle points. High variance in the update direction → noisy convergence path.

**Mini-batch SGD (the practical default):** compute gradient on a small batch (e.g., 32–4096 samples). Balances the accuracy of full-batch and the speed/regularizing-noise of pure SGD; also enables efficient parallelization on GPUs (vectorized batch operations).

**Momentum:** accumulates a moving average (exponentially weighted) of past gradients and uses that to update parameters, instead of just the current gradient:
```
v_t = β * v_{t-1} + (1-β) * ∇J(θ_t)        θ_{t+1} = θ_t - η * v_t
```
Intuition: like a ball rolling downhill picking up speed — it dampens oscillations in directions where the gradient keeps flipping sign (e.g., across a narrow ravine) and accelerates in directions with a consistent gradient. `β` (often 0.9) controls how much history is retained.

**RMSprop:** adapts the learning rate **per parameter** by dividing by a running average of the squared gradients:
```
s_t = β * s_{t-1} + (1-β) * (∇J)²           θ_{t+1} = θ_t - η * ∇J / (√s_t + ε)
```
Intuition: parameters with consistently large gradients get their effective learning rate shrunk (to avoid overshooting); parameters with small/rare gradients get an effectively larger learning rate (to speed up learning on sparse/rare features). Solves the problem of a single global learning rate being wrong for different parameters.

**Adam (Adaptive Moment Estimation)** — the default optimizer for deep learning, including LLM pretraining: **combines momentum + RMSprop.** Tracks both the first moment (mean of gradients, like momentum) and second moment (uncentered variance of gradients, like RMSprop), with bias-correction terms because early estimates are biased toward zero at initialization:
```
m_t = β1*m_{t-1} + (1-β1)*g_t                    (momentum term)
v_t = β2*v_{t-1} + (1-β2)*g_t²                    (RMSprop term)
m̂_t = m_t / (1-β1^t)     v̂_t = v_t / (1-β2^t)     (bias correction)
θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)
```
Typical defaults: `β1=0.9, β2=0.999, ε=1e-8`. Adam converges fast and needs less learning-rate tuning than plain SGD, which is why it's the default for training transformers (often as **AdamW**, which decouples weight decay from the gradient update — fixes a subtle bug where L2 regularization interacts poorly with Adam's adaptive scaling).

**Cross-questions:**
- *"Why is SGD noise sometimes good?"* — It acts as an implicit regularizer and can help the optimizer escape sharp, narrow minima (which tend to generalize worse) in favor of wider, flatter minima (which tend to generalize better).
- *"Why AdamW over Adam for transformers?"* — Standard Adam applies weight decay through the gradient (`g = g + λθ`), which gets scaled by the adaptive learning rate — meaning parameters with large gradient history get *less* effective regularization, which is unintended. AdamW applies weight decay directly to the weights, decoupled from the gradient-based update.
- *"What's a saddle point and why does it matter in high dimensions?"* — A point where gradient is zero but it's not a min/max (min in some directions, max in others). In high-dimensional loss landscapes (like deep nets), saddle points are far more common than local minima and can slow down plain gradient descent; momentum-based methods help escape them.

<a name="p1-7"></a>
### 7. Loss Functions

A loss function measures how wrong a single prediction is; the **cost function** is typically the average loss over the dataset (terms are often used interchangeably in interviews — don't worry about being pedantic unless asked directly).

- **Logistic loss (log loss)** — used in logistic regression, derived from negative log-likelihood of a Bernoulli distribution:
  ```
  L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
  ```
  Heavily penalizes confident-but-wrong predictions (as `ŷ→0` while `y=1`, `-log(ŷ)→∞`).

- **Cross-entropy loss** (general/multi-class form) — measures the difference between two probability distributions, the true one-hot label distribution and the predicted distribution:
  ```
  L = -Σ_k y_k * log(ŷ_k)          (only the true class term survives since y_k=0 elsewhere for one-hot labels)
  ```
  This is the standard classification loss for neural networks (paired with softmax output). **Memorize this formula** — it's asked constantly.

- **Hinge loss (SVM):**
  ```
  L = max(0, 1 - y*(w·x+b))          where y ∈ {-1, +1}
  ```
  Zero loss once a point is correctly classified **and** beyond the margin (`y*(w·x+b) ≥ 1`); loss grows linearly for points inside the margin or misclassified. This is what gives SVM its "sparse" solution — only support vectors (points with loss > 0, i.e., on/inside the margin) affect the final boundary.

- **Squared error / MSE (regression):** `L = (y - ŷ)²` — penalizes large errors quadratically, differentiable everywhere, but sensitive to outliers.
- **MAE (regression):** `L = |y - ŷ|` — robust to outliers (linear penalty), but not differentiable at 0 and gradient magnitude doesn't shrink near the optimum (can cause oscillation near convergence).
- **Huber loss:** quadratic for small errors, linear for large errors — combines MSE's smoothness near zero with MAE's robustness to outliers; has a threshold hyperparameter `δ`.

**Cross-question: "Why cross-entropy over MSE for classification?"** MSE + sigmoid is non-convex and its gradient can **vanish** when predictions are very wrong (saturated sigmoid region) — the model learns slowly exactly when it needs to learn fastest. Cross-entropy's gradient w.r.t. the pre-sigmoid logit simplifies to `(ŷ - y)`, which stays large when the prediction is very wrong — no vanishing-gradient problem, and it's derived directly from MLE, giving it a clean probabilistic interpretation.

---

<a name="p1-8"></a>
### 8. Feature Selection & Importance

**Why feature selection matters:** reduces overfitting (fewer irrelevant/noisy dimensions), speeds up training/inference, improves interpretability, and helps avoid the curse of dimensionality.

**Three families of methods:**
1. **Filter methods** — score each feature independently of any model, using a statistical measure, then keep the top-k. Examples: correlation with target, Chi-squared test (categorical), mutual information, variance threshold (drop near-constant features). Fast, model-agnostic, but ignores feature interactions.
2. **Wrapper methods** — use the actual downstream model's performance to evaluate feature subsets. Examples: Recursive Feature Elimination (RFE — repeatedly train, drop the least important feature, repeat), forward/backward stepwise selection. More accurate (accounts for interactions + the specific model) but computationally expensive (many retrains).
3. **Embedded methods** — feature selection happens *as part of* model training. Examples: **L1/Lasso regression** (drives some weights to exactly zero — automatic feature selection), tree-based **feature importance** (see below).

**Feature importance in tree ensembles:**
- **Mean Decrease in Impurity (MDI/Gini importance):** sum up how much each feature reduces impurity (Gini/entropy/variance) every time it's used for a split, averaged across all trees. Fast (computed during training) but **biased toward high-cardinality features** (features with many unique values get more chances to create a "lucky" split).
- **Permutation importance:** shuffle one feature's values (breaking its relationship with the target) and measure how much model performance drops on a held-out set. More reliable/less biased than MDI, but computationally more expensive (needs re-scoring).
- **SHAP values** (worth mentioning in 2026 interviews): game-theoretic approach (Shapley values from cooperative game theory) that fairly attributes a prediction's deviation from the average to each feature, even capturing interaction effects and giving both global and per-prediction (local) explanations.

---

<a name="p1-9"></a>
### 9. Model Evaluation & Metrics

**The confusion matrix (memorize this cold):**

|                     | Predicted Positive | Predicted Negative |
|---------------------|---------------------|---------------------|
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

- **Accuracy** = `(TP+TN) / (TP+TN+FP+FN)` — fraction correct overall. **Misleading on imbalanced data**: if 99% of data is negative, predicting "always negative" gives 99% accuracy while being useless.
- **Precision** = `TP / (TP+FP)` — "of everything I predicted positive, how much was actually positive?" High precision = few false alarms. Matters when **false positives are costly** (e.g., spam filter flagging real emails, or a fraud model blocking legitimate transactions).
- **Recall (Sensitivity, TPR)** = `TP / (TP+FN)` — "of everything actually positive, how much did I catch?" High recall = few missed cases. Matters when **false negatives are costly** (e.g., cancer screening, fraud detection — missing a real case is worse than a false alarm).
- **Specificity (TNR)** = `TN / (TN+FP)` — "of everything actually negative, how much did I correctly reject?"
- **F1-score** = harmonic mean of precision & recall = `2*P*R / (P+R)`. Harmonic mean (not arithmetic) is used because it punishes extreme imbalance between P and R more heavily — a model with precision=1.0 and recall=0.01 gets an arithmetic mean of ~0.5 (looks OK) but F1 ≈ 0.02 (correctly reflects it's terrible). **F-beta** generalizes this: `F_β` weighs recall `β` times as important as precision — use `β>1` (e.g., F2) when recall matters more (medical screening), `β<1` (F0.5) when precision matters more.

**Why precision/recall over just TPR when talking about "precision vs TPR"?** TPR (=recall) only looks at the positive class's actual instances and ignores how many false alarms you generated; **precision explicitly measures false-alarm rate relative to your positive predictions**, which is what stakeholders often actually care about (e.g., "of the accounts you flagged as fraud, how many really were fraud?" — that's precision, and it directly impacts operational cost of investigating false alarms).

**Choosing a metric for imbalanced datasets:** accuracy is a trap. Use precision/recall/F1, or better, **PR-AUC** (Precision-Recall AUC) which is more informative than ROC-AUC when the positive class is rare, because ROC-AUC can look deceptively good on imbalanced data (since TNR/FPR calculations are dominated by the abundant negative class).

**ROC curve:** plots **TPR (y-axis) vs. FPR (x-axis)** as you sweep the classification threshold from 0 to 1. `FPR = FP/(FP+TN) = 1 - Specificity`. A perfect classifier hugs the top-left corner (TPR=1, FPR=0); a random classifier is the diagonal line. **Threshold selection**: pick the threshold based on the business trade-off you need (e.g., Youden's J statistic `= TPR - FPR` maximization, or a fixed acceptable FPR).

**AUC (Area Under the ROC Curve):** a single number summarizing the ROC curve — probabilistically, it's **the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative example**. AUC=0.5 → random guessing; AUC=1.0 → perfect ranking. Good for **comparing models independent of a specific threshold**, but again, can be overly optimistic on heavily imbalanced data — check PR-AUC too.

**Multi-class extension:** compute confusion matrix as a `K×K` grid; precision/recall/F1 computed **per class** (one-vs-rest), then aggregated via:
- **Macro-average**: unweighted mean across classes — treats all classes equally regardless of size, good when you care about rare classes as much as common ones.
- **Micro-average**: aggregate TP/FP/FN globally then compute — dominated by large classes, equivalent to accuracy in the single-label multi-class case.
- **Weighted average**: like macro, but weighted by class support (frequency) — balances between the two.

---

<a name="p1-10"></a>
### 10. Cross-Validation & Model Selection

**Why cross-validation?** A single train/test split gives a noisy estimate of generalization performance (depends on which rows randomly ended up in test). Cross-validation reduces this variance by testing on multiple different splits and averaging.

**k-fold cross-validation:**
1. Split data into `k` equal-sized folds.
2. For each fold `i`: train on the other `k-1` folds, evaluate on fold `i`.
3. Average the `k` performance scores → final estimate (report mean ± std for a sense of variance too).

**What's a good `k`?** Common choices: `k=5` or `k=10`.
- Larger `k` → each training set is bigger (closer to using the full dataset) → **lower bias** in the performance estimate, but training happens `k` times so it's more expensive, and folds are more correlated with each other (**higher variance** across folds, since test sets are small and overlap a lot in what they exclude).
- Smaller `k` → faster, but each training set is smaller (more pessimistic bias) and test folds are larger (more stable per-fold estimate, but fewer of them to average).
- **Leave-One-Out CV (LOOCV)** = `k=n` (extreme case): nearly unbiased but very high variance and very expensive; generally avoided for large datasets.
- `k=10` is the most common default rule-of-thumb, balancing bias/variance/cost reasonably.

**Stratified k-fold**: for classification with imbalanced classes, ensures each fold preserves the overall class distribution — prevents a fold from accidentally having zero examples of a rare class.

**Time-series cross-validation**: never randomly shuffle time-ordered data — use **forward-chaining/rolling-origin** splits (train on past, test on future) to avoid leaking future information into training.

**Why do we need train/validation/test (3-way split) and not just train/test?**
- **Train**: fit model parameters.
- **Validation**: tune hyperparameters and select among models — if you tune hyperparameters using the test set, you leak information and your test performance becomes an overly optimistic estimate of true generalization (you've implicitly "fit" to the test set through repeated hyperparameter search).
- **Test**: touched exactly once, at the very end, to report an unbiased estimate of real-world performance.
- With cross-validation, you can fold validation into the CV loop (e.g., nested CV: inner loop for hyperparameter tuning, outer loop for unbiased performance estimate) and just keep a separate held-out test set.

<a name="p1-11"></a>
### 11. Unsupervised Learning: Clustering

**k-means clustering (centroid-based):**
1. Choose `k` (number of clusters), initialize `k` centroids (randomly, or better: **k-means++** which spreads initial centroids apart to speed convergence and avoid bad local minima).
2. **Assignment step:** assign each point to its nearest centroid (Euclidean distance).
3. **Update step:** recompute each centroid as the mean of points assigned to it.
4. Repeat 2-3 until assignments stop changing (convergence) or max iterations reached.

This is essentially **coordinate descent on the within-cluster sum of squares (WCSS/inertia)**: `Σ_k Σ_{x∈C_k} ||x - μ_k||²`. Guaranteed to converge (WCSS monotonically decreases each iteration) but only to a **local** minimum — that's why we run it multiple times with different initializations and keep the best result.

**Choosing k:** the **elbow method** (plot WCSS vs. k, look for the "elbow" where adding more clusters stops giving big improvements) or the **silhouette score** (measures how similar a point is to its own cluster vs. other clusters, ranges -1 to 1, higher is better; can be averaged to compare different k values objectively — more principled than the elbow method).

**Limitations of k-means:** assumes spherical, similarly-sized clusters (struggles with elongated or unevenly-sized clusters); sensitive to outliers (mean is not robust); requires choosing `k` upfront; sensitive to feature scaling (**always standardize features before k-means** since it's distance-based).

**Hierarchical clustering (connectivity-based):**
- **Agglomerative (bottom-up, more common):** start with every point as its own cluster; repeatedly merge the two closest clusters until only one remains (or a stopping criterion is met). Produces a **dendrogram** (tree diagram) that you can "cut" at any height to get a chosen number of clusters — a big advantage: no need to pick `k` upfront.
- **Divisive (top-down):** start with one cluster containing everything, recursively split. Less common (computationally expensive).
- **Linkage criteria** (how to measure distance between two *clusters*, not just points): **single linkage** (min distance between any pair — can create long "chained" clusters), **complete linkage** (max distance — tends toward compact clusters), **average linkage** (mean distance), **Ward's linkage** (merges clusters that minimize the increase in total within-cluster variance — often gives the most balanced, k-means-like clusters).

**DBSCAN (density-based):** clusters are regions of high density separated by regions of low density. Parameters: `ε` (neighborhood radius) and `minPts` (minimum points to form a dense region).
- A point is a **core point** if it has ≥ `minPts` neighbors within `ε`.
- A point is a **border point** if it's within `ε` of a core point but doesn't itself have enough neighbors.
- A point is **noise/outlier** if neither.
- Clusters form by connecting core points that are density-reachable from each other.

**Why DBSCAN over k-means?** Doesn't require specifying `k` upfront, can find **arbitrarily shaped clusters** (not just spherical), and naturally identifies **outliers/noise** as a byproduct rather than forcing every point into a cluster. Downsides: struggles with clusters of varying density (a single global `ε` doesn't fit all clusters), and is sensitive to the `ε`/`minPts` choice.

**Cross-question: "k-means vs KNN — don't confuse them!"** k-means is **unsupervised clustering** (finds groups, no labels). K-Nearest Neighbors (KNN) is a **supervised** classification/regression algorithm (given a labeled dataset, classify a new point by majority vote / average of its `k` nearest labeled neighbors — it's non-parametric, "lazy learning" with no explicit training phase, all computation happens at inference time). They share "k" and "distance" in their names but solve completely different problems.

---

<a name="p1-12"></a>
### 12. GMM, LSA, HMM

**Gaussian Mixture Models (GMM):** a **soft/probabilistic** version of clustering. Assumes data is generated from a mixture of `k` Gaussian distributions, each with its own mean `μ_k`, covariance `Σ_k`, and mixing weight `π_k`. Unlike k-means (hard assignment, each point belongs to exactly one cluster), GMM gives **soft assignments** — a probability that each point belongs to each cluster. Fit via **Expectation-Maximization (EM)**:
- **E-step:** given current parameters, compute the probability ("responsibility") that each Gaussian generated each point.
- **M-step:** given those responsibilities, update each Gaussian's `μ_k, Σ_k, π_k` to maximize the (weighted) likelihood.
- Iterate E/M until convergence (log-likelihood stops improving). Like k-means, converges to a local optimum → run multiple initializations.

GMM is more flexible than k-means because Gaussians can have different shapes/orientations (via full covariance matrices) rather than being forced spherical, and it gives calibrated uncertainty (soft membership) rather than a hard boundary.

**Latent Semantic Analysis (LSA):** an NLP technique — applies **SVD (Singular Value Decomposition)** to a term-document matrix (rows=terms, columns=documents, entries=TF-IDF or counts) to find a **lower-dimensional "latent semantic" space** where words/documents with similar meaning end up close together, even if they never literally co-occur (captures synonymy). Predecessor to modern word embeddings; mostly superseded by neural embeddings/transformers today, but still comes up as a "classic NLP" question testing whether you understand dimensionality reduction applied to text.

**Hidden Markov Models (HMMs):** model a sequence of **observed** outputs generated by a sequence of **hidden (latent) states** that evolve according to Markov dynamics (next state depends only on current state — the "Markov property").
- **Transition probability**: `P(state_t | state_{t-1})` — probability of moving from one hidden state to another.
- **Emission probability**: `P(observation_t | state_t)` — probability of seeing a given observed output given the hidden state.
- Classic use cases: part-of-speech tagging, speech recognition, gene sequencing.
- **Viterbi algorithm [Advanced]:** a dynamic-programming algorithm that finds the **single most likely sequence of hidden states** given a sequence of observations (as opposed to just the probability of the observation sequence, which is computed by the simpler Forward algorithm). It works by keeping, at each time step and each possible state, the best score of the most likely path ending in that state, and backtracking once you reach the end — O(states² × sequence length) instead of the naive exponential brute force over all possible state sequences.
- Note for 2026 context: HMMs have been largely displaced by neural sequence models (RNNs/transformers) for most tasks, but the *concepts* (latent state, transition dynamics, dynamic programming over sequences) still show up conceptually (e.g., beam search in decoding is a cousin of Viterbi-style search).

---

<a name="p1-13"></a>
### 13. Dimensionality Reduction (PCA, ICA, t-SNE)

**Principal Component Analysis (PCA):** finds a new set of orthogonal axes ("principal components") ordered by how much **variance** of the data they capture. PC1 = direction of maximum variance; PC2 = direction of next-most variance, orthogonal to PC1; and so on.

**Mechanism:**
1. Center the data (subtract the mean; often also standardize/scale each feature to unit variance if features are on different scales).
2. Compute the **covariance matrix** of the features.
3. Compute its **eigenvectors and eigenvalues** (or equivalently, run SVD on the centered data matrix directly — more numerically stable in practice).
4. Eigenvectors = principal component directions; eigenvalues = amount of variance explained by each component.
5. Project data onto the top `k` eigenvectors (sorted by eigenvalue, descending) to get a `k`-dimensional representation.

**Why it's useful:** reduces dimensionality while preserving as much variance (information) as possible, removes multicollinearity between features (new axes are uncorrelated by construction), speeds up downstream models, helps visualization (project to 2-3D), and can denoise data (dropping low-variance components often drops noise).

**Choosing `k`:** look at the **explained variance ratio** (eigenvalue_i / sum of all eigenvalues), often via a **scree plot**, and keep enough components to explain e.g. 90-95% of total variance.

**Key limitation:** PCA is **linear** — it can't capture nonlinear manifold structure (a kernel-PCA or autoencoder can). Also: PCA is unsupervised, so the directions of max variance might not align with what's useful for a downstream classification task (that's what LDA is for, if you have labels — see section 3).

**Independent Component Analysis (ICA):** unlike PCA (which finds *uncorrelated*, variance-ranked components — a linear/2nd-order-statistics notion), ICA finds components that are **statistically independent** (a stronger condition using higher-order statistics, not just uncorrelated) and specifically seeks **non-Gaussian** latent sources. Classic use case: the "cocktail party problem" — given several microphones recording a mix of overlapping voices, ICA can separate out the individual, independent speech signals (blind source separation). Where PCA answers "what are the axes of maximum variance," ICA answers "what are the underlying independent generative sources."

**t-SNE (t-distributed Stochastic Neighbor Embedding):** a **nonlinear** dimensionality reduction technique, used almost exclusively for **visualization** (typically to 2D/3D), not as a general preprocessing step for downstream models. Mechanism (high level):
1. Compute pairwise similarities between points in the **high-dimensional** space (using a Gaussian kernel — probability that point `i` would pick point `j` as its neighbor).
2. Compute pairwise similarities in the **low-dimensional** embedding using a **Student's t-distribution** (heavier tails than Gaussian — this is the "t" in t-SNE, and it's a deliberate choice to combat the "crowding problem," giving more room for moderately-distant points to be placed further apart in low-D).
3. Optimize the low-dimensional point positions (via gradient descent) to minimize the **KL divergence** between the high-D and low-D similarity distributions.

**Key facts/gotchas (frequently cross-questioned):** t-SNE preserves **local structure** (nearby points stay nearby) much better than global structure — **inter-cluster distances and cluster sizes in a t-SNE plot are NOT meaningfully interpretable** (a common mistake). It's stochastic (different runs → different layouts, unless you fix the random seed) and has a key hyperparameter, **perplexity**, that roughly controls the balance between local and global structure (effectively, an estimate of the number of neighbors each point should have) — results can look very different across perplexity values, so it's good practice to try a few. Computationally expensive for large datasets (though approximations like Barnes-Hut and openTSNE help); **UMAP** is a common modern alternative that's faster and better preserves some global structure.

**PCA vs t-SNE vs UMAP (cross-question):** PCA = linear, fast, deterministic, preserves global variance structure, good general-purpose preprocessing. t-SNE = nonlinear, slow, stochastic, great for visualization of local cluster structure, NOT for preprocessing before another model (embeddings aren't stable/meaningful for that). UMAP = nonlinear, faster than t-SNE, better global structure preservation, increasingly preferred for both visualization and as an actual preprocessing step.

<a name="p1-14"></a>
### 14. Bias-Variance Tradeoff & Regularization

**The core idea:** the expected test error of a model can be decomposed as:
```
Expected Test Error = Bias² + Variance + Irreducible Error
```
- **Bias**: error from wrong assumptions in the model (e.g., fitting a straight line to curved data). High bias → the model is too simple → **underfitting** (poor performance on both train and test).
- **Variance**: error from the model being too sensitive to the specific training set — small changes in training data cause large changes in the fitted model. High variance → the model is too complex/flexible → **overfitting** (great train performance, poor test performance — it memorized noise, not signal).
- **Irreducible error**: noise inherent in the data itself (e.g., measurement error) — no model can eliminate this.

**The tradeoff:** as model complexity increases (more features, deeper trees, more parameters), bias tends to decrease but variance tends to increase — that's why it's a *trade-off*, not something you can minimize both of freely. The goal is the complexity "sweet spot" that minimizes total error.

**How to detect overfitting/underfitting in practice:**
- **Underfitting**: training error is high (and validation error is similarly high/close to it).
- **Overfitting**: training error is low, but validation/test error is significantly higher — the gap between train and val performance is the tell.
- Plotted as a **learning curve** (error vs. training set size, or vs. training epochs) — this is a favorite interview visual to reason about.

**How to prevent overfitting (a checklist you should be able to rattle off):**
1. **Get more training data** (most effective if feasible — reduces variance directly).
2. **Regularization** (L1/L2 — see below), dropout (neural nets), early stopping.
3. **Reduce model complexity** (fewer features/parameters, shallower trees, prune).
4. **Ensemble methods** (bagging specifically targets variance reduction).
5. **Cross-validation** for proper hyperparameter selection (not to prevent overfitting directly, but to *detect* it reliably).
6. **Data augmentation** (especially in deep learning/vision — artificially expand the effective dataset).
7. **Feature selection** (remove noisy/irrelevant features).

**Regularization — L1 (Lasso) vs L2 (Ridge):** both add a penalty term to the loss function to discourage large weights, which reduces model complexity/variance:
```
Ridge (L2):  J(w) = MSE(w) + λ * Σ w_j²
Lasso (L1):  J(w) = MSE(w) + λ * Σ |w_j|
```
- **L2/Ridge**: penalizes the *square* of weights → shrinks all weights toward zero smoothly, but rarely exactly to zero. Good when you believe most/all features are somewhat useful and you mainly want to control magnitude (multicollinearity handling — it stabilizes `(XᵀX + λI)⁻¹`, fixing the invertibility issue from section 2). Has a closed-form solution.
- **L1/Lasso**: penalizes the *absolute value* → because of the shape of the L1 penalty (a diamond in weight-space vs. L2's circle), optimization tends to hit corners of the constraint region where some weights are **exactly zero** — this gives **automatic, sparse feature selection**. Good when you believe many features are irrelevant and want a sparse, interpretable model. No closed-form solution (need iterative methods like coordinate descent).
- **Elastic Net**: combines both (`λ1*Σ|w| + λ2*Σw²`) — gets sparsity from L1 while keeping L2's stability, especially useful when features are highly correlated (Lasso alone tends to arbitrarily pick one of several correlated features).
- **Geometric intuition (a favorite whiteboard question):** draw the elliptical contours of the unregularized loss function and the constraint region (diamond for L1, circle for L2) — the optimal solution is where the loss contour first touches the constraint boundary. Because the L1 diamond has sharp corners on the axes, the loss contour is more likely to first touch at a corner (where one or more coordinates = 0) than the smooth L2 circle.

**Why does regularization reduce variance?** By shrinking weights, it limits how much the model can swing to fit noise in any particular training set — smaller weights mean smaller sensitivity to small input/data perturbations, i.e., a smoother decision function.

---

<a name="p1-15"></a>
### 15. Sampling Techniques

- **Uniform (simple random) sampling:** every item in the population has an equal probability of being selected. Simple, unbiased, but can be inefficient for rare-event/imbalanced scenarios (a random sample from a 1000:1 imbalanced dataset barely captures the minority class).
- **Stratified sampling:** split the population into subgroups ("strata," e.g., by class label or a demographic attribute), then sample proportionally (or with a chosen ratio) from **within each stratum**. Ensures the sample's composition matches (or deliberately controls) the population's — critical for imbalanced classification (stratified train/test splits, stratified k-fold — see section 10) and for A/B tests where you want representative subgroups.
- **Reservoir sampling:** an algorithm for sampling `k` items uniformly at random from a **stream of unknown or very large length** (e.g., you can't fit the whole stream in memory, or don't know `n` in advance — think: sampling from a live data pipeline or an enormous log file). Mechanism: keep the first `k` items in a "reservoir." For each subsequent item `i` (`i > k`), generate a random index `j` in `[0, i]`; if `j < k`, replace `reservoir[j]` with the new item. This guarantees, at any point, every item seen so far has had an equal `k/i` probability of being in the reservoir — a classic "trick" algorithm that's a favorite in coding/data-structure interview rounds too.

**Cross-question: "Why not always just take a simple random sample?"** — Simple random sampling can, by chance, badly misrepresent small subgroups, especially with rare classes or small strata — stratified sampling removes that sampling variance by construction. Reservoir sampling solves a different problem entirely (streaming/unknown-size data), not representativeness.

---

<a name="p1-16"></a>
### 16. Handling Messy Data (Missing, Imbalanced, Drift)

#### Missing Data
**First, diagnose the missingness mechanism** (a favorite "do you actually understand this" cross-question):
- **MCAR** (Missing Completely At Random): missingness has nothing to do with any observed or unobserved value. Safe to drop or simply impute.
- **MAR** (Missing At Random): missingness depends on *other observed* variables (e.g., income missing more often for younger respondents, but not related to income itself once you control for age). Can be handled with model-based imputation using the observed variables.
- **MNAR** (Missing Not At Random): missingness depends on the *unobserved value itself* (e.g., people with very high income refuse to report it). Hardest case — imputation can introduce bias; may need domain modeling of the missingness process itself.

**Handling strategies:**
- **Deletion**: drop rows (listwise deletion, safe if MCAR and few rows affected) or drop columns (if a feature is missing too often to be useful).
- **Simple imputation**: fill with mean/median (numeric) or mode (categorical). Median is more robust to skew/outliers than mean. Fast but reduces variance artificially and can distort relationships between features.
- **Model-based imputation**: predict the missing value from other features (regression imputation, k-NN imputation, or iterative methods like **MICE** — Multiple Imputation by Chained Equations, which imputes each feature conditioned on the others, iterating multiple times, and often generates multiple imputed datasets to also capture imputation *uncertainty*).
- **Indicator/flag approach**: add a binary "was this value missing" feature alongside the imputed value — lets the model learn if missingness itself is informative (relevant especially in MAR/MNAR cases).
- **Algorithms that natively handle missing values**: XGBoost/LightGBM learn an optimal default split direction for missing values directly from training data loss reduction (see section 5) — no imputation needed.

#### Imbalanced Data
**Why it's a problem:** models trained on imbalanced data tend to be biased toward the majority class (since minimizing overall error/loss is dominated by getting the majority class right), and accuracy becomes a misleading metric (section 9).

**Handling strategies:**
- **Resampling:**
  - **Oversampling** the minority class: random duplication, or better, **SMOTE** (Synthetic Minority Oversampling Technique) — generates *synthetic* minority examples by interpolating between existing minority points and their nearest minority neighbors, rather than just duplicating (reduces exact-duplicate overfitting).
  - **Undersampling** the majority class: randomly drop majority examples. Risk: throws away potentially useful data, can hurt if majority class is not that large to begin with.
  - Often combined (e.g., SMOTE + undersampling) or ensembled (train multiple models each on a different balanced subsample, then aggregate — "EasyEnsemble"/"BalancedBagging").
- **Class weighting**: instead of resampling data, reweight the **loss function** to penalize minority-class errors more heavily (e.g., `class_weight='balanced'` in scikit-learn — weight inversely proportional to class frequency). Achieves a similar effect to oversampling without changing dataset size.
- **Threshold tuning**: instead of the default 0.5 cutoff, choose a classification threshold that better balances precision/recall for your use case (informed by the ROC/PR curve).
- **Metric choice**: use precision/recall/F1/PR-AUC instead of accuracy (see section 9) — this doesn't fix the model but ensures you're evaluating it honestly.
- **Anomaly-detection framing**: if the minority class is extremely rare (e.g., <0.1%, fraud detection), sometimes it's more effective to reframe the problem as anomaly/outlier detection (isolation forests, one-class SVM, autoencoder reconstruction error) rather than standard classification.

#### Data Distribution Shift
- **Covariate shift**: `P(x)` changes between train and test/production, but `P(y|x)` stays the same (e.g., a new user demographic starts using your product, but their behavior-to-label relationship is unchanged). Detect via comparing feature distributions (e.g., **Population Stability Index (PSI)**, KL divergence, or a simple "train a classifier to distinguish train vs. production samples" test — if it can, you have shift). Mitigate via importance weighting, or retraining on more recent/representative data.
- **Label/prior shift**: `P(y)` changes (e.g., fraud rate genuinely increases over time) but `P(x|y)` unchanged.
- **Concept drift**: the relationship `P(y|x)` itself changes over time (e.g., what "spam" looks like evolves as spammers adapt) — the hardest case, since the model's learned mapping is genuinely stale. Mitigate with continuous monitoring of live performance, periodic retraining on fresh data, or online-learning setups that adapt incrementally.
- **Practical monitoring in production**: track input feature distributions and prediction distributions over time (drift dashboards), track live performance against delayed ground truth when available, and set up alerts/automated retraining triggers.

---

## Part II — Deep Learning

<a name="p2-1"></a>
### 17. Feedforward Neural Networks & Backpropagation

**Structure:** a feedforward (fully-connected/dense) neural network is a stack of **layers**, each computing `a = f(Wx + b)`:
- **Input layer**: raw features.
- **Hidden layers**: each neuron computes a weighted sum of the previous layer's outputs, adds a bias, then applies a **nonlinear activation function**. Without nonlinearity, stacking layers would collapse into one big linear transformation (a composition of linear functions is still linear) — the nonlinearity is what gives neural nets their expressive power to approximate complex functions (Universal Approximation Theorem: a sufficiently wide single hidden layer with a nonlinear activation can approximate any continuous function, though depth is far more parameter-efficient in practice).
- **Output layer**: shape depends on the task — 1 neuron + sigmoid for binary classification, `K` neurons + softmax for multi-class, 1 (or more) linear neuron(s) for regression.

**Common activation functions:**
- **Sigmoid**: `σ(z)=1/(1+e^-z)`, range (0,1). Problem: **saturates** for large |z| (gradient → 0), causing **vanishing gradients** in deep nets — largely replaced by ReLU in hidden layers, but still used at output layers for binary probabilities.
- **Tanh**: range (-1,1), zero-centered (slightly better than sigmoid for hidden layers since gradients aren't all-positive), but still saturates.
- **ReLU** `max(0, z)`: the modern default. Doesn't saturate for `z>0` (no vanishing gradient there), computationally cheap, induces sparsity (many neurons output exactly 0). Downside: "**dying ReLU**" — if a neuron's weights update such that it always outputs ≤0, its gradient is permanently 0 and it stops learning. Fixes: **Leaky ReLU** (`max(αz, z)`, small slope `α` for negative inputs keeps gradient flowing), **GELU** (smooth approximation used in transformers, weights inputs by their percentile under a Gaussian — see also SwiGLU in Part IV), **ELU**, **Swish/SiLU**.
- **Softmax** (output layer, multi-class): `softmax(z)_k = e^{z_k} / Σ_j e^{z_j}` — converts a vector of raw scores ("logits") into a probability distribution over K classes that sums to 1.

**[EX] Activation function for classes that are NOT mutually exclusive (multi-label classification):** use **sigmoid independently on each output neuron** (not softmax!) — because softmax forces probabilities to sum to 1 across classes (mutually exclusive assumption: "this is exactly one of K things"), while independent sigmoids let each label be predicted with its own independent probability (e.g., an image can be tagged "outdoor" AND "dog" AND "daytime" simultaneously — each is a separate binary decision). Paired with **binary cross-entropy per label**, summed or averaged across labels.

**Backpropagation:** the algorithm that computes gradients of the loss with respect to every weight in the network, efficiently, using the **chain rule** applied backward through the computational graph.
1. **Forward pass**: compute and cache each layer's activations, layer by layer, ending in the loss.
2. **Backward pass**: starting from the loss, compute `∂L/∂output`, then propagate backward layer by layer using the chain rule: `∂L/∂W_l = ∂L/∂a_l * ∂a_l/∂z_l * ∂z_l/∂W_l`, reusing the gradients already computed for later layers (this reuse is exactly what makes it efficient — O(network size) instead of the exponential blowup of naively recomputing each partial derivative from scratch).
3. **Update**: apply gradient descent (or Adam, etc.) using these computed gradients.

**Why it matters that it's "just the chain rule + dynamic programming":** interviewers often want to hear you can explain *why* it's efficient (shared sub-computation, like memoization) rather than just "it computes gradients." Also worth knowing: modern frameworks (PyTorch/TensorFlow) implement this via **automatic differentiation**, building a computational graph at runtime (or compile time) and applying the chain rule mechanically node-by-node.

---

<a name="p2-2"></a>
### 18. RNNs, BPTT, Vanishing/Exploding Gradients

**Why RNNs?** Feedforward nets assume fixed-size, independent inputs. Sequential data (text, time series, audio) has **variable length** and **order-dependent** structure — a Recurrent Neural Network processes a sequence one step at a time, maintaining a **hidden state** `h_t` that acts as a running summary/memory of everything seen so far:
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```
The same weight matrices (`W_hh`, `W_xh`, `W_hy`) are **shared/reused at every time step** — this is what makes it "recurrent" and lets it handle arbitrary-length sequences with a fixed number of parameters.

**Backpropagation Through Time (BPTT):** to train an RNN, you conceptually "unroll" it across all time steps into an equivalent deep feedforward network (one "layer" per time step, sharing weights), then apply standard backpropagation across this unrolled graph. Because the same weight matrix is reused at every step, the gradient w.r.t. `W_hh` is the **sum of contributions from every time step**, each involving a **product of Jacobians** going back through time (chain rule across many steps).

**The vanishing/exploding gradient problem:** because BPTT's gradient involves repeatedly multiplying the same recurrent weight matrix (and activation derivatives) across many time steps — roughly `∂h_T/∂h_0 = Π_{t} W_hh * diag(tanh'(z_t))` — this product can either:
- **Vanish** (shrink toward 0) if the repeated factors have magnitude < 1, meaning gradients from far in the past become negligibly small — the network effectively can't learn long-range dependencies (can't connect an early word to a much-later word).
- **Explode** (grow toward ∞) if the factors have magnitude > 1, causing unstable, huge weight updates (loss can spike to NaN).

**Why exploding is "easier" to fix:** **gradient clipping** (rescale the gradient vector if its norm exceeds a threshold) is a simple, effective, commonly-used fix for exploding gradients. Vanishing gradients are the harder, more fundamental problem — the network structurally can't propagate signal, so it needs an **architectural fix**, which is exactly why LSTMs (and later, attention/transformers) were invented.

---

<a name="p2-3"></a>
### 19. LSTM & Dropout in Sequence Models

**LSTM (Long Short-Term Memory):** an RNN variant specifically designed to solve the vanishing gradient problem by introducing a **cell state** `C_t` — a separate "memory highway" that information can flow along largely unchanged, plus a system of **gates** (small neural nets with sigmoid outputs, values in (0,1), acting as "how much to let through") that control what gets added, removed, or read from that memory:

- **Forget gate** `f_t = σ(W_f·[h_{t-1}, x_t] + b_f)`: decides what fraction of the old cell state to discard.
- **Input gate** `i_t = σ(W_i·[h_{t-1}, x_t] + b_i)` and candidate values `C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)`: decides what new information to add to the cell state.
- **Cell state update**: `C_t = f_t * C_{t-1} + i_t * C̃_t` — this is the key equation. Notice it's an (mostly) **additive**, gated update, not a repeated matrix multiplication like vanilla RNN's `h_t`. Gradients flowing back through this additive path don't get repeatedly multiplied by the same weight matrix at every step, so they can flow much further back without vanishing (as long as the forget gate stays near 1 for relevant information).
- **Output gate** `o_t = σ(W_o·[h_{t-1}, x_t] + b_o)`, and hidden state `h_t = o_t * tanh(C_t)`: decides what part of the (updated) cell state to expose as the output/hidden state at this time step.

**Why does this fix vanishing gradients?** The cell state's update is essentially `C_t = f_t*C_{t-1} + (new stuff)` — an additive recurrence, not a multiplicative one — so `∂C_T/∂C_0` involves a **sum/product of forget gates** rather than repeated multiplication by a fixed weight matrix and squashed activation derivative. If the network learns to keep forget gates close to 1 for important long-term information, gradients can flow across many time steps nearly unimpeded — this is the core architectural insight ("gradient superhighway").

**Does the vanishing gradient problem fully go away in LSTMs?** No — it's *mitigated*, not eliminated (forget gates can still shrink gradients if they're not near 1, and very long sequences can still be hard) — this nuance is a good cross-question answer that shows real understanding rather than a memorized "LSTMs fix vanishing gradients" soundbite.

**Applying dropout to LSTMs — why it's tricky:** naive dropout (randomly zeroing activations each forward pass, independently at each time step) applied to the **recurrent connections** (`h_{t-1}→h_t`) is destructive — it injects noise into the memory pathway at every single time step, and since the *same* hidden state is reused/propagated across the whole sequence, this compounds and disrupts the network's ability to retain long-term memory. The standard fix, **Variational/Recurrent Dropout**, uses the **same dropout mask at every time step** within a given forward pass/sequence (rather than resampling the mask at each step) — so the *same* subset of units is dropped consistently through time, which lets the recurrent state still carry consistent information without inconsistent noise being injected fresh at every step. Dropout is typically only applied to the **non-recurrent** connections (input-to-hidden, hidden-to-output) in the simplest/original schemes, with the recurrent-dropout variants extending it more carefully to recurrent connections too.

---

<a name="p2-4"></a>
### 20. Seq2Seq & Attention

**Sequence-to-Sequence (Seq2Seq) models:** designed for tasks where input and output are both sequences, possibly of different lengths (machine translation, summarization). Classic architecture (pre-Transformer):
- **Encoder** (an RNN/LSTM): reads the entire input sequence and compresses it into a single fixed-size **context vector** (typically the encoder's final hidden state).
- **Decoder** (another RNN/LSTM): initialized with that context vector, generates the output sequence one token at a time, feeding each generated token back in as input for the next step (autoregressive generation).

**The bottleneck problem:** forcing the *entire* input sequence's information through a single fixed-size vector is a severe information bottleneck, especially for long sequences — the decoder has to "remember" everything about a 50-word sentence from one vector, and quality degrades noticeably as sequence length grows. This directly motivated **attention**.

**Attention (the fix):** instead of relying solely on one fixed context vector, let the decoder, **at every output step**, look back at **all** encoder hidden states and compute a weighted combination of them — where the weights ("attention scores") are dynamically computed based on how relevant each input position is to the current output step being generated.

Mechanism (general attention, e.g., Bahdanau/Luong style):
1. Compute an **alignment score** between the decoder's current state and each encoder hidden state (via a small learned function, or simply a dot product).
2. **Softmax** those scores into weights that sum to 1 — this is the attention distribution.
3. Compute a **weighted sum ("context vector") of encoder hidden states**, weighted by the attention distribution — a different context vector is computed fresh at every decoding step.
4. Combine this context vector with the decoder's own hidden state to produce the output token.

**Self-attention** (the key generalization that led to Transformers): instead of attention between two *different* sequences (encoder→decoder, "cross-attention"), self-attention lets a sequence attend **to itself** — every position computes attention weights over every other position **in the same sequence**, letting each token directly gather context from any other token, regardless of distance, in a single step (O(1) path length between any two positions, vs. an RNN's O(sequence length) path length). This is what removed the need for recurrence entirely and is the foundation of the Transformer (next section).

---

<a name="p2-5"></a>
### 21. Transformer Architecture — Full Deep Dive
*(This is one of the single most-tested topics in 2026 ML interviews — "explain the transformer in detail, no kidding" — be ready to draw the block diagram and write out every equation.)*

**High-level architecture:** the original Transformer (Vaswani et al., "Attention Is All You Need") is an **encoder-decoder** stack, though today most LLMs use a **decoder-only** variant (GPT-style) — know both, but be ready to focus on decoder-only since that's what powers modern LLMs.

**Step 1 — Input embeddings + positional encoding:** tokens are converted to dense vectors via an embedding lookup table, then a **positional encoding** is added (since, unlike RNNs, self-attention has *no* inherent notion of order — it treats input as a *set*, not a sequence — so position must be injected explicitly; see Part IV §27 for modern approaches like RoPE).

**Step 2 — Scaled Dot-Product Self-Attention (the core mechanism):**
For each token, project its embedding into three vectors via learned weight matrices: **Query (Q)**, **Key (K)**, **Value (V)**:
```
Q = X·W_Q     K = X·W_K     V = X·W_V
Attention(Q,K,V) = softmax( QKᵀ / √d_k ) · V
```
**Intuition:** think of it as a soft, differentiable lookup/retrieval system. The **Query** represents "what am I looking for" (from this token's perspective). The **Key** represents "what do I contain" (from every other token's perspective, including itself). `QKᵀ` computes a **similarity/relevance score** between every query and every key (dot product = high when vectors point in a similar direction). Softmax turns those scores into a probability distribution (attention weights summing to 1) over all positions. The **Value** vectors are what actually gets aggregated — the output for a token is a weighted average of all tokens' Value vectors, weighted by how relevant each token's Key was to this token's Query.

**Why scale by `√d_k`? (a near-guaranteed cross-question)** As the dimensionality `d_k` of Q/K vectors grows, the dot products `QKᵀ` grow in magnitude too (variance of a dot product of random vectors scales with dimension). Very large dot-product values pushed into softmax cause it to saturate — the softmax output becomes extremely peaked (near one-hot), and in that saturated regime, the **gradient of softmax becomes vanishingly small**, stalling learning. Dividing by `√d_k` renormalizes the dot products back to unit variance (roughly), keeping softmax in a well-behaved regime with healthy gradients — it's purely a **numerical stability / gradient health** fix, not a change to what's representable.

**Multi-Head Attention:** instead of computing one attention function on the full-dimensional Q/K/V, **split into `h` heads**, each operating on a smaller `d_k = d_model/h` dimensional subspace, computing attention independently and in parallel, then concatenating the results and passing through a final linear projection. **Why?** A single attention head is forced to average over one single "type" of relationship. Multiple heads let the model jointly attend to information from **different representation subspaces at different positions simultaneously** — e.g., empirically, different heads often specialize (one head tracks syntactic dependencies, another tracks coreference, another tracks positional/local patterns) — giving richer, more diverse relational modeling than one head could alone, at roughly the same total compute (each head is smaller).

**Step 3 — Residual connections + Layer Normalization:** each sub-layer (attention, and later the feedforward block) is wrapped as `x + Sublayer(Norm(x))` (pre-norm, the modern standard — see Part IV §29 for pre-norm vs post-norm discussion). **Residual/skip connections** let gradients flow directly through addition (identity path) even through very deep stacks, mitigating vanishing gradients in deep networks (same core idea as ResNets) and letting each layer learn a *refinement* to its input rather than having to reconstruct the input's information from scratch. **Layer Normalization** normalizes activations across the feature dimension (as opposed to Batch Norm's normalization across the batch dimension) — this matters because Batch Norm's statistics depend on batch composition, which is problematic for variable-length sequences and small/varying batch sizes; LayerNorm normalizes each individual token's vector independently of other tokens/batch elements, which is much better suited to sequence models and works identically at inference time regardless of batch size.

**Step 4 — Position-wise Feedforward Network (FFN):** applied identically and independently to each position's vector after attention — typically `Linear → nonlinearity (originally ReLU, modern LLMs use GELU/SwiGLU) → Linear`, usually expanding to a larger intermediate dimension (e.g., 4x `d_model`) then projecting back down. **Why is this needed if attention already mixes information across tokens?** Attention is fundamentally a **linear, weighted-averaging** operation over Value vectors (given fixed weights) — it moves/mixes information *between* positions but doesn't itself introduce nonlinear transformation of a token's own representation. The FFN provides the actual **nonlinear per-token computation/transformation** capacity (this is where a lot of the model's "knowledge" is thought to be stored, per interpretability research on MLP layers as key-value memories).

**Full block (repeated N times):** `x → x + SelfAttention(Norm(x)) → x + FFN(Norm(x))` — stack this block N times (e.g., N=96 for GPT-3-scale models) to build the full network.

**Encoder vs. Decoder differences:**
- **Encoder** (e.g., BERT): **bidirectional** self-attention — every token can attend to every other token, including ones that come *after* it (no masking) — good for understanding/representation tasks (classification, embeddings), bad for generation (would be "cheating" by seeing the future).
- **Decoder** (e.g., GPT): **causal (masked) self-attention** — a token can only attend to itself and **earlier** tokens; a mask sets attention scores for future positions to `-∞` before the softmax (so they become exactly 0 after softmax) — this enforces the autoregressive property needed for next-token-prediction generation.
- **Cross-attention** (in encoder-decoder models like the original Transformer or T5): the decoder additionally attends to the encoder's output representations (Q from decoder, K/V from encoder) — this is how the decoder "looks at" the source sequence while generating, in translation-style tasks.
- **Decoder-only** (GPT, Llama, Claude-style architectures): no separate encoder at all — the entire input+output is treated as one sequence with causal masking throughout; this has become the dominant architecture for general-purpose LLMs because of its simplicity and strong scaling properties.

**End-to-end summary you should be able to say out loud in ~90 seconds:** "Tokens are embedded and given positional information. Each layer first runs multi-head self-attention, where every token computes a Query, Key, and Value; attention weights are the softmax of scaled Query-Key dot products, and the output is a weighted sum of Values — this lets each token gather context from anywhere in the sequence in a single step, unlike an RNN's sequential bottleneck. That's followed by a position-wise feedforward network for nonlinear per-token processing. Both sub-layers are wrapped in residual connections and layer normalization for stable, deep training. Stack this block N times. Decoder-only models like GPT add a causal mask so tokens can only attend to the past, enabling autoregressive next-token generation."

**Recommended resource:** [The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/) — read this before any interview where transformers might come up; it's the most commonly recommended visual walkthrough.

---

<a name="p2-6"></a>
### 22. Embeddings

**What is an embedding?** A dense, low(er)-dimensional vector representation of a discrete object (a word, token, item, user, image patch) such that **semantically/functionally similar objects end up close together in vector space** (by some distance/similarity metric, usually cosine similarity or dot product). Contrast with **one-hot encoding**, which is sparse, high-dimensional (vocab-size length), and treats every pair of distinct items as equally (maximally) dissimilar — one-hot vectors carry zero notion of similarity, while embeddings *learn* similarity structure from data.

**Word embeddings (classic, pre-transformer NLP):**
- **Word2Vec**: trains a shallow neural net on a proxy task to *produce* embeddings as a byproduct. Two variants: **CBOW** (predict a target word from its surrounding context words) and **Skip-gram** (predict surrounding context words from a target word — generally works better for rare words / smaller datasets). Learns embeddings such that words appearing in similar contexts end up with similar vectors (distributional hypothesis: "a word is characterized by the company it keeps"). Famous property: learned vector arithmetic captures analogies, e.g., `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`.
- **GloVe**: builds embeddings from **global co-occurrence statistics** (a word-word co-occurrence matrix across the whole corpus) rather than local context windows — combines ideas from matrix-factorization methods (like LSA, section 12) with the local-context intuition of Word2Vec.
- **Limitation of both**: **static** embeddings — each word gets exactly one vector regardless of context, so "bank" (river) and "bank" (financial) get the same embedding. This is exactly what motivated **contextual embeddings**.

**Contextual embeddings (modern/transformer-based):** models like BERT/GPT produce a **different embedding for the same word depending on its surrounding context**, because the embedding at each layer is the *output of self-attention*, which by definition mixes in information from the surrounding tokens. This is a major leap in quality for anything downstream (retrieval, classification, RAG) because it captures polysemy and nuanced meaning-in-context.

**How embeddings are used today (know this for LLM interviews):** the **token embedding table** at an LLM's input layer (lookup: token ID → vector) is trained jointly with the rest of the model. Downstream, "embedding models" (e.g., sentence/document embedding models used in RAG pipelines) typically take a pretrained transformer and extract/pool a fixed-size vector (e.g., mean-pooling over token outputs, or a dedicated `[CLS]`-style token) tuned (often via contrastive learning — pulling similar text pairs' embeddings together, pushing dissimilar pairs apart) specifically to make **semantic similarity ≈ vector similarity**, which is exactly the property retrieval/RAG systems rely on.

---

## Part III — Statistical ML & Misc

<a name="p3-1"></a>
### 23. Bayesian Methods: Naive Bayes, MAP, MLE

**Bayes' Theorem (the foundation):**
```
P(A|B) = P(B|A) * P(A) / P(B)
```
In ML terms: `P(class | features) = P(features | class) * P(class) / P(features)` — i.e., **posterior ∝ likelihood × prior**.

**Naive Bayes classifier:** applies Bayes' theorem with one strong ("naive") simplifying assumption — **all features are conditionally independent given the class label**:
```
P(y | x_1,...,x_n) ∝ P(y) * Π_i P(x_i | y)
```
This turns an otherwise intractable joint distribution estimation problem into `n` easy, independent univariate estimation problems (just estimate `P(x_i | y)` for each feature separately — e.g., counts/frequencies for categorical features, or fit a Gaussian per class for continuous features → "Gaussian Naive Bayes"). Prediction: pick the class `y` that maximizes this posterior.

**Why does it work well despite the independence assumption being almost always false in practice?** (a very common cross-question) Because Naive Bayes only needs to get the **ranking** of posterior probabilities across classes correct (i.e., `argmax` over classes), not the exact probability values — even with correlated features violating the independence assumption, the errors introduced often affect all classes' probability estimates in a correlated way that doesn't flip which class has the highest score. It's also extremely fast, needs little training data, and is a strong, cheap baseline — classic use case: spam filtering, text classification (with word-presence/count features).

**Maximum Likelihood Estimation (MLE):** a general method for estimating model parameters `θ` by finding the value that **maximizes the likelihood of the observed data**:
```
θ_MLE = argmax_θ  P(data | θ) = argmax_θ  Π_i P(x_i | θ)
```
In practice, we maximize the **log-likelihood** instead (turns the product into a sum — numerically stable and easier to differentiate, and log is monotonic so the argmax is unchanged): `θ_MLE = argmax_θ Σ_i log P(x_i|θ)`.
**Key fact:** minimizing cross-entropy loss (for classification) or squared error (for regression, under a Gaussian-noise assumption) is **mathematically equivalent to MLE** — this is a very common "aha, connect the dots" interview question: "show that minimizing MSE is the same as MLE under Gaussian noise" (because the Gaussian's log-likelihood, when you drop constants, reduces exactly to `-(y-ŷ)²` up to scaling).

**Maximum A Posteriori (MAP) estimation:** extends MLE by incorporating a **prior** belief over the parameters `θ`, via Bayes' theorem:
```
θ_MAP = argmax_θ  P(θ | data) = argmax_θ  P(data|θ) * P(θ)
```
**MAP vs. MLE:** MLE only looks at the likelihood of the data; MAP additionally weighs in a prior over what parameter values are plausible *before* seeing data. **Key connection to regularization** (another favorite "connect the dots" question): if you place a **Gaussian prior** on the weights in linear/logistic regression and derive the MAP estimate, you get **exactly L2/Ridge regularization** — the prior's negative log translates into the `λΣw²` penalty term. A **Laplace prior** on the weights similarly gives you **exactly L1/Lasso regularization**. This shows regularization isn't an arbitrary "add a penalty" hack — it's principled Bayesian reasoning about plausible parameter values, and it also explains intuitively *why* L1's Laplace prior (which has a sharp peak at zero) induces sparsity while L2's Gaussian prior (smooth, no sharp peak) doesn't.

---

<a name="p3-2"></a>
### 24. Statistical Significance: R², p-values

**R² (coefficient of determination):** measures the proportion of variance in the target variable that's explained by the model, relative to a trivial baseline (predicting the mean every time):
```
R² = 1 - (SS_res / SS_tot)
SS_res = Σ(y_i - ŷ_i)²        (residual sum of squares — your model's error)
SS_tot = Σ(y_i - ȳ)²          (total sum of squares — error of predicting the mean)
```
`R²=1` means perfect prediction; `R²=0` means your model is no better than just predicting the mean; **R² can go negative** if your model is *worse* than predicting the mean (a classic gotcha fact interviewers like to check — many people wrongly assume R² is bounded at [0,1]). **Caveat**: R² mechanically increases (or stays the same) as you add more features/predictors, even useless ones — this is why **Adjusted R²** exists, which penalizes for the number of predictors, giving a fairer comparison across models with different feature-set sizes.

**p-values:** in the context of hypothesis testing (e.g., testing whether a regression coefficient is significantly different from zero, or whether a treatment effect in an A/B test is real), the p-value is **the probability of observing a result at least as extreme as what you actually observed, assuming the null hypothesis is true** (e.g., null hypothesis: "this feature has no real relationship with the target / this coefficient is truly 0"). A small p-value (conventionally `< 0.05`) suggests the observed effect would be unlikely under the null, giving evidence *against* the null hypothesis.

**Common misinterpretations to explicitly avoid saying in an interview** (this is exactly what gets probed): a p-value is **NOT** the probability that the null hypothesis is true, and it's **NOT** the probability that your result happened by chance. It's a statement about how surprising your data would be *if* the null were true — a subtly different, frequentist conditional probability. Also: **statistical significance ≠ practical/business significance** — with a large enough sample size, even a tiny, practically meaningless effect can become "statistically significant" (p<0.05); always look at effect size alongside the p-value.

---

<a name="p3-3"></a>
### 25. Outliers & Similarity Metrics

**Detecting outliers:**
- **Statistical methods**: Z-score (`|x - μ|/σ > threshold`, typically 2-3), IQR method (flag points below `Q1 - 1.5*IQR` or above `Q3 + 1.5*IQR`) — simple, work well for roughly-Gaussian, single-feature data.
- **Model-based**: **Isolation Forest** (randomly partitions data via random splits — outliers, being "different," tend to get isolated into their own leaf in far fewer splits than normal points, so **average path length to isolate a point** becomes an anomaly score), **Local Outlier Factor (LOF)** (compares a point's local density to its neighbors' local density — a point in a much sparser neighborhood than its neighbors is flagged as an outlier; good for detecting *local* anomalies that a single global threshold would miss), **one-class SVM**, **DBSCAN**'s noise label (section 11), autoencoder reconstruction error (points that reconstruct poorly are likely anomalous/out-of-distribution).
- **What to do once found**: depends on cause — remove if it's a data-entry error, cap/winsorize (clip to a percentile) if it's a legitimate-but-extreme value that would otherwise dominate a distance-/scale-sensitive model, transform the feature (e.g., log-transform) to reduce its influence, or use robust models/losses (Huber loss, MAE, tree-based models which are naturally robust to outliers since splits only depend on ordering, not magnitude).

**Similarity / dissimilarity metrics:**
- **Euclidean distance**: `√Σ(x_i - y_i)²` — straight-line ("as the crow flies") distance. Sensitive to feature scale (always standardize first) and can behave poorly in high dimensions ("curse of dimensionality" — distances between all pairs of points tend to converge/become less discriminative as dimensionality grows).
- **Manhattan distance (L1/city-block)**: `Σ|x_i - y_i|` — sum of absolute coordinate differences ("grid-based" movement). More robust to outliers in individual dimensions than Euclidean (no squaring), often preferred in high-dimensional/sparse settings.
- **Cosine similarity**: `(x·y) / (||x|| * ||y||)` — measures the **angle** between two vectors, ignoring their magnitude. This is exactly why it's the standard choice for **text/embedding similarity**: two documents about the same topic might have very different lengths (hence different magnitude in a bag-of-words/embedding sum), but if their *direction* in vector space is similar, cosine similarity correctly identifies them as similar regardless of that magnitude difference — this is the single most important practical fact to know for RAG/embedding-search interview questions.
- **Mahalanobis distance [advanced]**: like Euclidean distance, but accounts for the **covariance structure** of the data — `√((x-y)ᵀ Σ⁻¹ (x-y))` where `Σ` is the covariance matrix. Effectively "whitens" the space first (accounting for correlated/differently-scaled features) before measuring distance, so it correctly identifies a point as an outlier relative to the data's actual (possibly elongated, correlated) distribution shape, whereas Euclidean distance would treat all directions equally and potentially miss a point that's unusual only *given the correlation structure* between features.

---

## Part IV — LLMs & Foundation Models (2026)
*Interviewer framing to expect: "LLMs are now expected breadth, not a specialty" — treat everything below as core knowledge, not a bonus topic.*

<a name="p4-1"></a>
### 26. Attention at Scale: MHA → MQA → GQA → MLA

Standard **Multi-Head Attention (MHA)** (Part II §21) gives every head its own full set of Query, Key, **and** Value projections. This is expressive, but at inference time it becomes a serious **memory bandwidth bottleneck** because of the KV cache (§28) — every head needs its own K and V cached for every token, and moving all that data between GPU memory and compute units, not the compute itself, is what limits generation speed at long context/large batch.

- **MHA (Multi-Head Attention)**: `h` heads, each with its own Q, K, V. Best quality, largest KV cache (proportional to `h`).
- **MQA (Multi-Query Attention)**: all `h` query heads **share a single K and V head**. Drastically shrinks the KV cache (by a factor of `h`) and speeds up inference, but quality can degrade somewhat since all heads are forced to read from the same K/V representation.
- **GQA (Grouped-Query Attention)**: a middle ground — query heads are split into `g` groups, and each group shares one K/V head (so `g` K/V heads total, `1 < g < h`). This is the practical sweet spot adopted by **Llama 2/3** and most modern open LLMs: it captures most of MQA's memory/speed savings while recovering most of MHA's quality, because you're not forcing *all* heads down to one shared K/V, just clusters of them.
- **MLA (Multi-Head Latent Attention, DeepSeek):** a further, more aggressive optimization — instead of caching full K/V vectors per head, MLA **compresses K and V into a low-rank latent vector** that's cached instead (much smaller), then reconstructs the per-head K/V via a learned up-projection at attention time. This achieves KV-cache savings competitive with or better than GQA while preserving quality closer to full MHA, because the compression is *learned* (jointly trained to minimize information loss) rather than simply *sharing* raw K/V across heads.

**Why did Llama adopt GQA specifically (a named cross-question)?** Because MQA's quality drop was noticeable at larger model scales, while full MHA's KV cache became prohibitively expensive at the longer context lengths and larger batch sizes needed for practical serving — GQA was empirically found to recover nearly all of MHA's quality while retaining most of MQA's inference speed/memory benefits, making it the best quality-per-byte-of-KV-cache tradeoff at the time.

---

<a name="p4-2"></a>
### 27. Positional Encodings: RoPE, ALiBi, Long Context

Self-attention has no inherent sense of token order (it's permutation-invariant over positions) — position must be injected explicitly.

- **Absolute/learned positional embeddings** (original Transformer, early GPT/BERT): add a fixed (sinusoidal) or learned vector per absolute position index to the token embedding. Simple, but **doesn't generalize well beyond the trained context length** — the model has never seen position 5000 during training if it only ever trained on sequences up to 2048, so it has no learned embedding/behavior for it.

- **RoPE (Rotary Position Embedding)** — the modern standard (Llama, most current open/closed LLMs): instead of *adding* a positional vector, RoPE **rotates** the Query and Key vectors by an angle proportional to their absolute position, in a way specifically designed so that the dot product `Q·K` between two positions ends up depending only on their **relative distance**, not their absolute positions. Why this matters:
  - It naturally encodes **relative position** directly into the attention computation itself (rather than relying on the model to *learn* relative relationships from absolute position embeddings), which tends to generalize better.
  - Because it's a rotation (a smooth, continuous, mathematically well-behaved operation), positions can be **interpolated/extrapolated** more gracefully to unseen lengths, which is exactly why RoPE-based models are the ones amenable to long-context extension tricks like **position interpolation** (compress/rescale position indices to "fit" a longer sequence into the range the model was trained on) and **YaRN** (a more sophisticated frequency-dependent rescaling of RoPE's rotation rates that better preserves both local and global attention patterns when extending context length far beyond training length).

- **ALiBi (Attention with Linear Biases)**: instead of modifying Q/K vectors, ALiBi directly **adds a fixed, distance-proportional penalty** to the attention score before softmax — the penalty grows linearly with the distance between two tokens (with a different slope per head), so far-apart tokens get systematically down-weighted. Simple and cheap, and was shown to extrapolate to longer sequences reasonably well without any fine-tuning, though RoPE (often combined with extension tricks) has become the more widely adopted default in top-performing 2025-2026 LLMs.

**Cross-question: "Why are rotary embeddings preferred over learned positional embeddings?"** Because (1) they encode relative position directly into the attention mechanism rather than as a separate additive signal the model must learn to interpret, (2) they generalize to longer sequences much better via principled extrapolation methods (YaRN, NTK-aware scaling), and (3) they don't add extra learned parameters (the rotation is a deterministic function of position), keeping the model simpler.

---

<a name="p4-3"></a>
### 28. KV Cache & FlashAttention

**The problem KV cache solves:** in autoregressive decoding, generating token `t+1` requires attending over all previous tokens `1...t`. Naively, this means **recomputing the Key and Value vectors for every previous token, at every single generation step** — for a sequence of length `n`, that's O(n²) redundant work overall, since each of the `n` tokens' K/V gets recomputed `n` times.

**The fix — KV caching:** since a token's Key and Value vectors (at a given layer) don't change once computed (they only depend on that token and everything before it, which is fixed once generated), you can **compute them once and cache them**. At each new decoding step, you only need to compute Q/K/V for the **single new token**, then attend against the cached K/V of all prior tokens. This turns each generation step into O(n) work (attending against `n` cached vectors) instead of O(n²) total — i.e., **linear cost per new token** rather than recomputing everything from scratch.

**Why KV cache dominates memory at long context / large batch (a key 2026 systems question):** cache size scales as `2 (K and V) × num_layers × num_heads × head_dim × sequence_length × batch_size × bytes_per_element`. At long context lengths (100K+ tokens) and/or large batch sizes (many concurrent requests being served), this can **exceed the memory footprint of the model's weights themselves** — this is precisely the motivation behind MQA/GQA/MLA (§26, shrink the per-token cache size) and KV-cache quantization (§35, shrink the bytes per element).

**FlashAttention:** an **IO-aware, exact** (not approximate) reimplementation of the attention computation, designed around the fact that on modern GPUs, **moving data between slow HBM (high-bandwidth memory) and fast on-chip SRAM is the actual bottleneck**, not the raw FLOPs of the matrix multiplications. Standard attention implementations materialize the full `n×n` attention score matrix in HBM (read/write it multiple times across the softmax, masking, and value-weighting steps) — very memory-bandwidth-heavy, and its memory footprint is O(n²) in sequence length.

FlashAttention instead **tiles** the computation: it processes Q, K, V in blocks small enough to fit in fast SRAM, computes attention for that block, and uses a clever **online/running softmax** algorithm to incrementally combine results across blocks *without ever materializing the full n×n score matrix in HBM*. This reduces memory reads/writes dramatically (same exact mathematical result, just computed more IO-efficiently) — which is **the** reason training on long context sequences became practically feasible; without it, the O(n²) memory and bandwidth cost of attention would make long-context training prohibitively slow/memory-hungry even with the same total FLOPs.

**Cross-question: "Is FlashAttention an approximation?"** No — it's mathematically **exact**, same output as standard attention, bit-for-bit equivalent (up to floating point summation order). It's purely a systems/IO optimization, not an approximation algorithm — an important distinction many candidates get wrong.

---

<a name="p4-4"></a>
### 29. Modern Block Internals: RMSNorm, SwiGLU, Pre-Norm

**Pre-norm vs. post-norm:** the original Transformer applied LayerNorm **after** the residual addition ("post-norm": `x = LayerNorm(x + Sublayer(x))`). Modern LLMs (GPT-2 onward, Llama, etc.) apply it **before** the sublayer instead ("pre-norm": `x = x + Sublayer(LayerNorm(x))`). **Why the switch?** Post-norm training becomes unstable at large depth — gradients flowing through the residual stream get repeatedly renormalized/rescaled by each LayerNorm, which can destabilize training of very deep networks and typically requires a careful learning-rate warmup to avoid divergence early in training. Pre-norm keeps a **clean, unimpeded residual/gradient highway** (the raw `x` passes straight through the addition, untouched by normalization), which is dramatically more stable for training very deep transformer stacks — at the (minor) cost of slightly reduced final representation quality/capacity compared to a *successfully-trained* post-norm model, which is a trade modern large-scale training happily accepts for reliability.

**RMSNorm (Root Mean Square Normalization):** a simplified variant of LayerNorm used in Llama and most modern LLMs. Standard LayerNorm normalizes by both **subtracting the mean and dividing by the standard deviation** (re-centering AND re-scaling). RMSNorm **drops the mean-centering step** and only rescales by the root-mean-square of the activations:
```
RMSNorm(x) = x / √(mean(x²) + ε) * γ         (γ = learned scale parameter)
```
**Why it's preferred:** empirically, the re-centering (mean subtraction) step in LayerNorm contributes relatively little to performance, but computing it still costs extra computation (an extra pass to compute the mean, an extra subtraction). RMSNorm gets **similar training stability/performance with less compute** — a small but real efficiency win multiplied over billions of forward passes at LLM scale.

**SwiGLU / GeGLU activations (the modern FFN):** the original Transformer's FFN used `ReLU(xW1)W2`. Modern LLMs use **gated linear units** instead, where a "gate" (from a nonlinear branch) elementwise-multiplies a separate linear branch:
```
SwiGLU(x) = (Swish(xW) ⊙ xV) W2          where Swish(z) = z*σ(z)  (a.k.a. SiLU)
GeGLU(x)  = (GELU(xW) ⊙ xV) W2
```
This adds an extra learned linear projection (`V`) compared to the plain ReLU FFN, effectively letting the network learn a smoother, **input-dependent gating** of which features pass through, rather than a fixed pointwise nonlinearity. Empirically (per the "GLU Variants Improve Transformer" paper and subsequent large-scale ablations), these gated variants consistently give better downstream quality for the same parameter/compute budget — which is why essentially every major 2024-2026 open LLM (Llama, Mistral, Qwen, etc.) uses SwiGLU as its FFN activation.

---

<a name="p4-5"></a>
### 30. Mixture-of-Experts (MoE)

**Core idea:** replace a single, dense FFN block (used identically for every token) with **many parallel "expert" FFNs**, and route each token to only a **small subset** (e.g., top-2) of experts via a learned **router/gating network**. This decouples **total parameter count** from **compute cost per token** — you can have a model with hundreds of billions of total parameters, but each individual token only activates (and pays the compute cost of) a small fraction of them.

**Mechanism:**
1. A lightweight router network computes a score for each expert given the current token's representation (typically a simple linear layer + softmax).
2. Select the **top-k** experts (commonly k=1 or k=2) by score.
3. Route the token through only those selected experts' FFNs, combine their outputs (weighted by the router's scores), and pass that forward — every other expert is skipped entirely for this token, saving compute.

**Active vs. total parameters (a key vocabulary distinction interviewers check):** "**total parameters**" = the sum of all experts' parameters plus shared components (attention layers, embeddings, etc.) — this is what determines the model's storage/memory footprint. "**Active parameters**" = the parameters actually used in the forward pass for a *given token* (shared components + only the selected top-k experts) — this is what determines the actual FLOPs/compute cost and, roughly, inference latency. E.g., **Mixtral 8x7B** has ~47B total parameters but only ~13B active per token (top-2 of 8 experts); **DeepSeek-V3** uses a much larger number of smaller, more fine-grained experts with a similarly large total-vs-active gap, plus shared experts that are always active alongside the routed ones.

**Load balancing (the key training challenge):** without intervention, the router tends to **collapse** — a few "popular" experts get routed to disproportionately often (a rich-get-richer dynamic, since experts that get more training signal early on get better, which makes the router prefer them even more), while other experts are starved of training data and stay undertrained/useless. The fix is an **auxiliary load-balancing loss** added during training that explicitly penalizes uneven routing distribution across experts, encouraging the router to spread tokens more evenly (recent approaches, e.g. DeepSeek's, use auxiliary-loss-free balancing via a learned per-expert bias term adjusted based on recent load, which avoids the accuracy-hurting side effects that a heavy-handed auxiliary loss can introduce).

**Why go through this complexity at all — what's the payoff?** MoE lets you scale total model capacity (and thus knowledge/quality) far beyond what would be affordable if every parameter had to be dense-computed for every token — you get GPT-4-class total capacity at a fraction of the per-token inference compute/cost of an equivalently-large dense model, which is exactly why most frontier-scale models in 2025-2026 (GPT-4-class, DeepSeek-V3, Mixtral, etc.) use MoE architectures.

---

<a name="p4-6"></a>
### 31. Tokenization & Scaling Laws

**Tokenization:** the process of converting raw text into a sequence of discrete integer IDs (tokens) that the model actually operates on. Word-level tokenization (one token per whitespace-separated word) fails badly on rare words, typos, and morphologically rich languages (huge, sparse vocabulary; out-of-vocabulary words become impossible to represent). Character-level tokenization avoids OOV entirely but makes sequences extremely long (worse compute cost, and harder for the model to learn word-level semantics from tiny character-level units).

- **BPE (Byte-Pair Encoding)**: starts with a vocabulary of individual characters, then **iteratively merges the most frequent adjacent pair of tokens** into a new single token, repeating until the target vocabulary size is reached. Common words end up as single tokens; rare words get broken into meaningful sub-word pieces (e.g., "unhappiness" → "un" + "happi" + "ness"); truly novel character sequences can still be represented by falling back to smaller sub-pieces or characters. This gives a good balance: manageable vocabulary size, no true OOV problem, and reasonable sequence length.
- **Byte-level BPE** (used by GPT-2/3/4, most modern LLMs): operates on raw UTF-8 **bytes** rather than unicode characters as the base vocabulary — guarantees the tokenizer can represent **any** input (any language, emoji, arbitrary binary-ish text) with a small, fixed base vocabulary of 256 possible byte values, with BPE merges then building up common multi-byte sequences into larger tokens on top of that.
- **SentencePiece**: a tokenizer *library/framework* (used by many models, e.g., Llama, T5) that treats text as a raw stream of Unicode characters (including spaces, via a special marker like `▁`) rather than pre-splitting on whitespace first — this makes it **language-agnostic** (works cleanly for languages that don't use whitespace to separate words, like Chinese/Japanese, unlike whitespace-pretokenized BPE) and fully reversible/lossless (can exactly reconstruct the original text, including spacing, from the token sequence).
- **Vocabulary size trade-off:** larger vocabulary → shorter sequences for the same text (fewer tokens needed, since more gets packed into single tokens) → less compute spent on attention (which scales with sequence length) but a **larger embedding table and output softmax layer** (more parameters/compute at the input and output projection specifically) — typical modern LLM vocab sizes range from ~32K to 250K+.
- **Context window**: the maximum number of tokens a model can process in one forward pass (attend over) — determined by what the model was trained/architected for; extending it after pretraining generally requires the long-context techniques discussed in §27 (position interpolation, YaRN) plus continued/fine-tuning training on longer sequences.

**Scaling laws (Chinchilla / compute-optimal training):** empirically, a model's loss follows a smooth, predictable power-law relationship with three things: **model size (parameters, N)**, **training data size (tokens, D)**, and **compute budget (FLOPs, C ≈ 6ND)**. The key finding from DeepMind's "Chinchilla" paper: earlier large models (like GPT-3) were significantly **undertrained relative to their size** — for a fixed compute budget, prior practice over-invested in parameter count and under-invested in training tokens. Chinchilla's compute-optimal recipe: **scale model size and training tokens roughly in proportion** (specifically, roughly ~20 training tokens per parameter was their empirical compute-optimal ratio) — meaning a smaller model trained on proportionally more data can match or beat a larger model trained on less data, for the *same* total compute cost. This reshaped how frontier labs allocate compute budgets, and is a very commonly tested concept ("what did the Chinchilla paper show, and why does it matter?").

**Emergent abilities:** capabilities (e.g., multi-step arithmetic, chain-of-thought reasoning, certain forms of in-context learning) that appear to show up somewhat **abruptly** at a certain model/data scale, rather than improving smoothly and predictably along with loss — a much-debated topic (some research argues these "emergent jumps" are partly an artifact of using discontinuous/thresholded evaluation metrics rather than evidence of a truly discontinuous underlying capability curve; worth mentioning both sides if asked, since it's a genuinely contested empirical question in the field).

---

<a name="p4-7"></a>
### 32. Training Pipeline Overview

Modern LLM training is a **multi-stage pipeline**, not one monolithic training run. Know the stages, in order, and what each one is *for*:

**1. Pretraining** — self-supervised **next-token prediction** on web-scale text corpora (trillions of tokens: web crawl, books, code, papers). Objective: given tokens `1...t-1`, predict token `t`, minimizing cross-entropy loss averaged over every position in every sequence. No human labels required — the "label" at each position is just the next token in the raw text itself. This is where the vast majority of compute/cost goes, and it's what gives the model its broad world knowledge, language ability, and reasoning substrate. The output of this stage is often called a **"base model"** — it can complete text plausibly but has no notion of "being helpful," following instructions, or refusing harmful requests; it just continues whatever pattern it's shown.

**2. SFT (Supervised Fine-Tuning) / instruction tuning** — fine-tune the base model on a (typically much smaller, curated) dataset of `(prompt → ideal response)` demonstrations, written or curated by humans (or increasingly, generated/filtered by other strong models). This teaches the model the **format and behavior** of being a helpful assistant that follows instructions, rather than just continuing arbitrary text patterns — same next-token-prediction loss, just applied to curated instruction-following data instead of raw web text, and typically only computing loss on the *response* tokens, not the prompt.

**3. Preference alignment (RLHF & alternatives)** — after SFT, the model can follow instructions but its outputs may not be reliably ranked by *quality/preference* — one response might be more helpful, honest, or safe than another equally "grammatically valid" response. This stage aligns the model to **human (or AI) preferences** using comparison data (humans or an AI judge rank multiple candidate responses to the same prompt) rather than single "correct" demonstrations — see §33 for the full menu of algorithms (RLHF/PPO, DPO, and modern alternatives) used at this stage.

**4. Reasoning RL** — a newer (2024-2026) stage layered on top, specific to building models that do extended chain-of-thought reasoning (OpenAI's o-series, DeepSeek-R1 style models): apply **RL with verifiable rewards** (correctness on math problems, passing unit tests for code, valid structured outputs) to directly optimize the model's reasoning *process*, rewarding it for reaching correct final answers via extended intermediate reasoning, rather than optimizing for "sounds good to a human rater" as in classic RLHF. See §33 (GRPO, RLVR) for the specific algorithms driving this.

**Why layer these stages instead of doing one big training run?** Each stage optimizes a fundamentally different objective, and doing them in this specific order matters: pretraining needs massive, cheap, unlabeled data to build broad capability; SFT needs a comparatively tiny amount of *very high-quality* curated demonstrations (quality over quantity) to teach behavior/format, which would be impossibly expensive to do at pretraining scale; preference/RL stages need comparison or verifiable-reward signal that's structurally different from either of the above and further refines behavior beyond what direct demonstration can teach (e.g., you can rank "which of these two responses is better" even when neither is a perfect gold-standard demonstration, which is far cheaper/more scalable than writing perfect demonstrations for every possible situation).

---

<a name="p4-8"></a>
### 33. Post-Training Zoo: SFT, RLHF/PPO, DPO, SimPO, KTO, ORPO, GRPO, RLVR

*This is the single highest-density "central 2026 interview theme" in this whole guide: picking the right post-training method for your data and compute constraints.* Modern production stacks **layer** these (SFT → preference optimization → RL) rather than picking just one.

| Algorithm | Data required | Extra models needed | Rel. cost | Use when |
|---|---|---|---|---|
| **SFT** | Curated instruction (prompt→response) demos | none | low | Teach format / instruction-following (always first) |
| **RLHF (PPO)** | Human preference labels | reward model + critic/value + reference | high | Classic; mostly displaced by simpler methods |
| **DPO** | Preference **pairs** (chosen vs rejected) | reference model | medium | Default offline alignment; trains like SFT, no RL loop |
| **SimPO** | Preference pairs | none (reference-free) | med-low | DPO without a reference model (avg-logprob implicit reward) |
| **KTO** | **Binary** thumbs up/down (no pairs) | none | low | Cheap / noisy feedback; only unpaired signal available |
| **ORPO** | Instruction demos only | none | low | Merge SFT + preference tuning into one stage |
| **GRPO** | Prompts only (sample a **group** of responses, group-relative advantage) | none (no critic) | medium | RL without a value net; reasoning / math (DeepSeek) |
| **RLVR** | Tasks with **verifiable** reward (unit tests, math answer, valid JSON) | automated verifier | medium | Code / math / tool-use where correctness is checkable |

**Walk through each in more depth (be ready to explain any of these on request):**

**RLHF with PPO (the "classic" pipeline):**
1. Collect human preference comparisons (given a prompt, humans rank/pick the better of two model responses).
2. Train a separate **reward model** to predict these human preferences (a classifier that outputs a scalar "quality score" for any response).
3. Use **PPO (Proximal Policy Optimization)**, an RL algorithm, to fine-tune the LLM (the "policy") to maximize the reward model's score — while a **KL-divergence penalty against a frozen reference model** (usually the SFT model) keeps the policy from drifting too far from sensible, fluent language (unconstrained reward maximization can lead to "reward hacking" — degenerate outputs that game the reward model without actually being good).
4. Requires *four* models in memory simultaneously during training: the policy (being trained), the reward model, a value/critic model (estimates expected future reward, used to reduce variance in the RL gradient estimate), and the frozen reference model — this is why RLHF/PPO is expensive and complex to get right (notoriously finicky to tune, sensitive to hyperparameters, high infrastructure/engineering burden).

**DPO (Direct Preference Optimization) — the big simplification:** DPO's key insight is that, mathematically, the RLHF objective (maximize reward, subject to a KL constraint from the reference model) has a **closed-form relationship** between the optimal policy and the reward function — meaning you can **substitute the reward model out entirely** and derive a loss that directly optimizes the policy on preference pairs, using simple supervised-learning-style gradient descent (no RL loop, no separate reward model, no critic/value network — just the policy and a frozen reference model for the KL term). This collapses RLHF's multi-model, multi-stage RL pipeline into a single, stable, supervised-loss training run on `(prompt, chosen, rejected)` triples — which is why DPO became the default "good enough, way simpler" choice for offline preference alignment.

**SimPO:** goes one step further than DPO by **removing the need for a reference model entirely** — instead of measuring preference relative to a frozen reference policy's log-probabilities (as DPO does), SimPO uses the **average log-probability of a response itself** (length-normalized) as an implicit reward signal. This saves the memory/compute of keeping a second reference model around during training and removes a source of potential mismatch between the reference model and the training data distribution.

**KTO (Kahneman-Tversky Optimization):** DPO/SimPO both need **paired** preference data (this response beats that response for the *same* prompt) — but often, real-world feedback is just unpaired binary signal (a single response got a 👍 or 👎, with no direct comparison collected). KTO is designed to work directly with this cheaper, noisier, **unpaired binary feedback**, drawing on prospect theory (humans weigh losses more heavily than equivalent gains) to shape its loss function — useful when preference-*pair* collection is impractical but you have plenty of thumbs-up/down style signal (e.g., mined from real user interactions).

**ORPO (Odds Ratio Preference Optimization):** merges what's normally two separate stages (SFT, then a preference-optimization stage) into **one single training run** — it adds a preference-based penalty term (based on the odds ratio between the chosen and rejected response's likelihoods) directly on top of the standard SFT loss, so a single training pass simultaneously teaches instruction-following *and* preference alignment. Attractive when you want to reduce total training stages/cost and have (prompt, chosen, rejected) data available from the start.

**GRPO (Group Relative Policy Optimization) — the reasoning-model workhorse (DeepSeek):** PPO's biggest cost/complexity driver is the separate **critic/value network** needed to estimate a baseline for computing the advantage (how much better was this action than expected) in the RL update. GRPO **removes the critic entirely**: for a given prompt, it samples a **group** of `k` responses from the current policy, computes a reward for each (via a reward model or, in RLVR settings, a verifiable/rule-based reward), and estimates each response's "advantage" simply as **how it compares to the group's average/normalized reward** — no separately-trained value network needed. This substantially lowers memory and compute overhead versus PPO (one fewer full model to train and hold in memory) while still providing a variance-reduced RL training signal, which is exactly why it's the algorithm behind DeepSeek's reasoning-model training and has become the go-to choice for **reasoning/math RL** at scale.

**RLVR (Reinforcement Learning with Verifiable Rewards):** rather than a *learned* reward model (which can be gamed/hacked, or simply wrong), RLVR uses **automated, rule-based/verifiable reward signals** — did the code pass its unit tests? Does the final numeric answer match the known correct answer? Is the output valid JSON matching a schema? This sidesteps reward-model imperfection/hacking entirely for domains where correctness genuinely **can** be checked programmatically (math, code, structured-output tasks, tool-use success) — it's often paired with GRPO as the RL algorithm, since together they form the "cheap-to-verify reward + no-critic-needed RL" combination that underlies most current reasoning-model training recipes.

**Other names worth knowing (can name-drop for depth):** **DAPO** — a set of stabilization techniques for very long chain-of-thought RL training (addressing issues like entropy collapse and reward signal noise that arise specifically when reasoning traces get very long). **RLAIF (RL from AI Feedback)** — same structure as RLHF, but the preference labels/rankings are generated by another (usually stronger) **AI model acting as judge**, rather than human annotators — massively cheaper and more scalable than human labeling, at the cost of inheriting whatever biases/blind-spots the judge model has.

**Key talking points to have ready (the exact phrases interviewers are listening for):**
- "**DPO collapses the reward model + RL loop into a single supervised loss on preference pairs.**"
- "**GRPO drops the critic network** — it estimates advantage via group-relative, normalized rewards across multiple sampled responses instead of a learned value function."
- "**GRPO/RLVR power reasoning models** — verifiable rewards (math, code) plus critic-free RL is the current recipe behind o-series/DeepSeek-R1-style models."
- "**Algorithm rankings are scale-dependent** — empirically, online RL methods can outperform offline methods like SimPO at small scale (~1.5B parameters), while the ranking can flip at larger scale (~7B) — there's no single universally-best post-training algorithm across all model sizes/data regimes, so this is an active empirical/engineering decision, not a solved theoretical question."

---

<a name="p4-9"></a>
### 34. PEFT: LoRA & QLoRA

**The problem PEFT solves:** full fine-tuning of a large model means updating **every** parameter — for a model with tens/hundreds of billions of parameters, this requires storing gradients and optimizer states (e.g., Adam needs 2 extra copies of every parameter for its moment estimates) for the *entire* model, which can require many times the memory of the base model weights alone. It's also expensive to store a separate full copy of the model for every downstream task/customer you fine-tune for, and full fine-tuning risks **catastrophic forgetting** (the model overwrites/forgets general pretrained capabilities while overfitting to the narrow fine-tuning task/dataset).

**Full fine-tuning vs. PEFT trade-off (be ready to state this directly):** full fine-tuning gives the most task-specific quality headroom (nothing is constrained) but costs the most compute/memory and risks forgetting; **PEFT (Parameter-Efficient Fine-Tuning)** freezes almost all of the original model and only trains a small number of additional/modified parameters — much cheaper (memory and compute), much smaller artifacts to store/ship per task, and inherently resistant to catastrophic forgetting (since the vast majority of original weights never change) — at the cost of a somewhat lower ceiling on achievable task-specific performance versus full fine-tuning, particularly for tasks that require substantially different behavior from the base model.

**LoRA (Low-Rank Adaptation):** based on the empirical/theoretical observation that the **weight update** needed to adapt a pretrained model to a new task tends to have a **low "intrinsic rank"** — i.e., even though the full weight matrix `W` might be huge (`d×d`), the *change* `ΔW` needed for good task adaptation can be well-approximated by a much lower-rank matrix. LoRA freezes the original weight matrix `W` entirely and injects a trainable **low-rank decomposition** alongside it:
```
W_new = W_frozen + ΔW = W_frozen + B·A
```
where `A` is `(d × r)` and `B` is `(r × d)`, with rank `r` chosen to be **much smaller** than `d` (e.g., r=8, 16, 64 vs. d in the thousands) — so the number of trainable parameters is `2*d*r` instead of `d²`, a massive reduction. `A` is typically initialized randomly, `B` initialized to zero (so `ΔW=0` at the start of training, meaning the model behaves identically to the base model before any LoRA training happens — a clean, stable starting point). Only `A` and `B` need gradients/optimizer states, drastically cutting training memory. At inference, `B·A` can even be **merged directly back into `W`** (since it's just matrix addition), meaning LoRA adds **zero extra inference latency** once merged — a major practical advantage over other PEFT methods like adapters that add extra sequential layers.

**QLoRA:** combines LoRA with **quantization** of the frozen base model — the base model's weights are stored/loaded in a low-precision format (commonly 4-bit, using a specialized format like NF4 — "NormalFloat4," designed to match the actual distribution of neural network weights better than a naive uniform 4-bit quantization) **while the LoRA adapter weights (A, B) are still trained in higher precision** (e.g., bfloat16). This means you can fine-tune a model that would ordinarily require far more GPU memory than you have, because the (frozen, non-trainable) bulk of the parameters sit compressed in memory the whole time, and only the small trainable LoRA matrices need full-precision gradient computation — this is precisely what made fine-tuning genuinely large (65B+) models feasible on a single consumer/prosumer-class GPU.

**Other PEFT methods worth naming:** **Adapters** (small trainable bottleneck feedforward layers inserted between existing frozen transformer layers — unlike LoRA, these do add inference latency since they're sequential, not mergeable). **Prefix/prompt tuning** (freeze the entire model, and instead learn a small set of continuous "virtual token" embeddings prepended to the input that steer the frozen model's behavior — extremely parameter-efficient but generally has a lower quality ceiling than LoRA for complex tasks).

**Key LoRA hyperparameters to know:** **rank `r`** (higher rank = more trainable capacity/expressiveness, but more parameters and higher risk of overfitting on small datasets — typical range 4-64), **alpha `α`** (a scaling factor applied to the LoRA update, `ΔW = (α/r)·B·A` — controls the effective magnitude/learning strength of the adaptation relative to the frozen weights; often set as a multiple of `r`, e.g., `α=2r`), learning rate (LoRA can often tolerate/benefit from a notably higher learning rate than full fine-tuning would, since it's optimizing far fewer parameters), and which weight matrices to apply LoRA to (commonly just the attention Q/V projections in the original paper, though applying it to all linear layers, including the FFN, is now common practice and often gives better results for a modest additional parameter cost).

---

<a name="p4-10"></a>
### 35. Inference & Serving: Quantization, Paged Attention, Speculative Decoding

**Quantization:** reduces the numerical precision used to store/compute model weights (and sometimes activations) — e.g., from 16-bit floating point down to 8-bit or 4-bit integers/floats — to shrink memory footprint and increase throughput (lower precision means more values fit in the same memory bandwidth, and some hardware computes low-precision ops faster).
- **PTQ (Post-Training Quantization)**: quantize an already-fully-trained model, with no (or minimal, calibration-only) retraining. Fast and cheap to apply, but can lose more accuracy than QAT, especially at very low bit-widths.
- **QAT (Quantization-Aware Training)**: simulate quantization's precision-reduction effects **during training** (typically via a "fake quantization" forward pass that rounds values but still allows gradients to flow, e.g., via the straight-through estimator), so the model's weights are actually optimized to be robust to the eventual quantization — generally yields better accuracy at very low bit-widths than PTQ, at the cost of requiring (re-)training.
- **INT8 / INT4 methods**: **GPTQ** (a PTQ method — quantizes weights layer-by-layer, using second-order/Hessian information to choose quantization values that minimize the resulting output error, processing weights in a way that compensates for earlier quantization errors within the same layer) and **AWQ (Activation-aware Weight Quantization)** (observes that only a small fraction of weight channels are actually "salient"/important, based on the magnitude of their corresponding activations, and preferentially preserves precision for those specific channels while quantizing the rest more aggressively) — both are popular PTQ techniques for pushing LLMs down to 4-bit with minimal quality loss.
- **FP8**: an 8-bit *floating-point* format (as opposed to INT8's fixed-point integer format) — floating point's dynamic range (via its exponent bits) handles the wide range of magnitudes in neural network activations/gradients more gracefully than INT8, making FP8 increasingly popular for both **training** (not just inference) on newer GPU hardware with native FP8 support, since it needs less aggressive calibration than INT8 to avoid accuracy loss.
- **KV-cache quantization**: quantizing the cached Key/Value vectors specifically (not just the model weights) — important because, as covered in §28, **the KV cache can exceed the size of the model weights themselves at long context/large batch**, so compressing it (e.g., to INT8 or INT4) can be an even bigger win for serving long-context workloads than weight quantization alone. The accuracy/latency trade-off generally: more aggressive quantization → faster, smaller, cheaper to serve, but higher risk of accuracy degradation, especially compounding across many layers/long sequences — the right operating point is workload- and model-dependent, found empirically via evaluation on the specific downstream task.

**Paged Attention & continuous batching (vLLM):** two systems techniques that dramatically improve LLM serving throughput.
- **Paged Attention**: inspired directly by **virtual memory paging in operating systems** — instead of requiring each request's KV cache to be stored in one large, contiguous block of GPU memory (which causes severe memory fragmentation and waste, since you must over-allocate for the *maximum possible* sequence length upfront), Paged Attention stores the KV cache in **fixed-size, non-contiguous "pages/blocks,"** with a lookup table mapping logical sequence positions to physical memory blocks — this eliminates fragmentation/over-allocation waste and lets memory be shared efficiently across requests (e.g., when multiple requests share a common prompt prefix).
- **Continuous batching** (a.k.a. dynamic/in-flight batching): traditional (static) batching processes a fixed batch of requests together and waits for **all** of them to finish generating before starting a new batch — meaning a request that finishes early sits idle, wasting GPU capacity, while a batch is bottlenecked by its single slowest/longest request. Continuous batching instead **swaps in new requests to fill freed-up slots the moment any request in the batch finishes**, keeping the GPU's batch consistently full and dramatically increasing overall throughput.
- **Prefix caching**: when multiple requests share an identical prompt prefix (e.g., the same system prompt across many user requests), the KV cache for that shared prefix can be **computed once and reused** across all of them, avoiding redundant recomputation — pairs very naturally with Paged Attention's block-based memory layout.

**Speculative decoding:** normally, generating each token requires one full forward pass through the (large, expensive) target model — strictly sequential, one token at a time. Speculative decoding speeds this up using a **small, cheap "draft" model** to propose several candidate tokens ahead (e.g., 4-8 tokens) very quickly, then the large "target" model verifies all of those candidate tokens **in a single parallel forward pass** (checking whether it would have generated the same tokens) — any tokens that match are accepted "for free" (no need to run the expensive model token-by-token for those), and generation only falls back to the slower step-by-step method starting from the first token where the draft and target disagree. This exploits the fact that a single forward pass over multiple tokens (parallel/parallelizable on a GPU) is much cheaper *per token* than multiple separate sequential forward passes — so as long as the cheap draft model's guesses are reasonably good (agree with the target model often enough), this yields a significant real-world speedup with **mathematically identical output distribution** to standard decoding from the target model (it's an exact, lossless acceleration technique, not an approximation, similar in spirit to FlashAttention being exact).

**Parallelism strategies (for models/serving too large for one GPU):**
- **Tensor parallelism**: split individual weight matrices/layers **across multiple GPUs** (e.g., different GPUs each hold a slice of the same attention/FFN layer's weights), requiring communication between GPUs within a single layer's forward pass — needs very fast interconnect (e.g., NVLink) since communication happens frequently, at every layer.
- **Pipeline parallelism**: split the model **by layers** across GPUs (e.g., GPU 1 holds layers 1-10, GPU 2 holds layers 11-20), passing activations between GPUs as data flows through the layer stack — less communication-frequent than tensor parallelism, but can suffer from "pipeline bubbles" (idle time while waiting for earlier stages) unless carefully scheduled/overlapped across multiple micro-batches.
- **Sequence parallelism**: splits computation **along the sequence dimension** (rather than across layers/weights) — particularly useful for reducing the activation memory burden of very long sequences, often combined with tensor parallelism.
- **Prefill vs. decode phases**: LLM inference has two distinct computational regimes — **prefill** (processing the entire input prompt at once, in parallel — compute-bound, since you're doing one big parallel matrix multiply over all prompt tokens simultaneously) and **decode** (generating output tokens one at a time, autoregressively — memory-bandwidth-bound, since each step is a tiny amount of compute but requires reading the *entire* model's weights and KV cache from memory for just one new token). Because these two phases have such different bottlenecks (compute-bound vs. memory-bandwidth-bound), serving systems increasingly **separate/specialize hardware or scheduling for prefill vs. decode** ("disaggregated serving") to optimize each phase independently rather than treating a request as one monolithic workload.

---

<a name="p4-11"></a>
### 36. Decoding Strategies & In-Context Learning

**Decoding (turning next-token probability distributions into actual generated text):**
- **Greedy decoding**: always pick the single highest-probability token at each step. Fast, deterministic, but can produce repetitive/bland text and is provably suboptimal for finding the overall highest-probability *sequence* (a locally-best choice at each step doesn't guarantee a globally-best full sequence — classic greedy-algorithm suboptimality).
- **Beam search**: maintains the top-`k` (`beam width`) most probable partial sequences at each step (instead of just the single best), expanding each and keeping only the overall top-`k` again — finds a higher-probability full sequence than greedy, common in translation/summarization where there's often one clear "correct" output. Less common for open-ended chat/creative generation, where it tends to produce dull, generic, overly "safe" text (probability-maximizing sequences in language often correlate with blandness/repetition, not quality/interestingness) — this is a classic, testable insight.
- **Temperature**: rescales the logits before softmax (`softmax(logits / T)`) — `T < 1` sharpens the distribution (more confident, more deterministic, closer to greedy as T→0), `T > 1` flattens it (more randomness/diversity, more likely to sample low-probability/creative-but-riskier tokens), `T = 1` leaves the distribution unchanged.
- **Top-k sampling**: restrict sampling to only the `k` highest-probability tokens at each step (renormalize their probabilities, sample from just that subset) — prevents sampling absurd, very-low-probability tokens ("long tail" garbage) while still allowing some randomness/diversity among the plausible candidates.
- **Top-p / nucleus sampling**: instead of a fixed count `k`, dynamically select the **smallest set of tokens whose cumulative probability exceeds `p`** (e.g., p=0.9), then sample from that set. This adapts to the *shape* of the distribution at each step — when the model is very confident (probability mass concentrated on few tokens), the nucleus is small (behaves close to greedy); when the model is uncertain (probability spread across many plausible tokens), the nucleus is larger (allows more diverse sampling) — generally considered a more principled/adaptive choice than fixed top-k, and is the default/most common sampling strategy in production chat systems today (often combined with a temperature setting too).
- **Repetition penalty**: directly penalizes (down-weights) tokens that have already appeared in the generated text so far, to combat degenerate repetition loops that can otherwise occur especially at low temperature/greedy decoding.
- **Structured/constrained decoding**: forces generated output to conform to a formal grammar or schema (e.g., valid JSON matching a specific schema, or a specific programming language's syntax) by **masking out (setting to -∞ before softmax) any token that would make the output invalid** at each step, guaranteeing well-formed structured output — critical for reliable tool-use/function-calling and structured-data-extraction applications.

**In-context learning (ICL):** the striking ability of large pretrained LLMs to learn/adapt to a new task **from examples provided directly in the prompt, at inference time, with no gradient updates/weight changes at all** — purely by conditioning the forward pass on the provided context.
- **Zero-shot**: no examples given, just an instruction/question — relies entirely on the model's pretrained/aligned general knowledge and instruction-following ability.
- **Few-shot**: a handful of `(input → output)` example pairs are included directly in the prompt before the actual query, letting the model infer the task's pattern/format from those examples alone.
- **Chain-of-thought (CoT) prompting**: instead of asking the model to jump straight to a final answer, prompt it to generate **intermediate reasoning steps** first ("let's think step by step," or via few-shot examples that themselves show worked-out reasoning) — this reliably improves performance on tasks requiring multi-step logical/arithmetic reasoning, because it gives the model "more forward-pass compute" (more tokens = more sequential computation) to work through the problem incrementally, rather than being forced to produce a complex answer in a single next-token-prediction leap.
- **Self-consistency**: sample **multiple independent chain-of-thought reasoning paths** for the same question (using temperature-based sampling to get diverse reasoning), then take a **majority vote** over their final answers — since different sampled reasoning paths that happen to make different (independent) mistakes are unlikely to all agree on the same wrong answer, while paths that reach the *correct* answer tend to converge more consistently, majority voting boosts accuracy over any single sampled chain of thought.
- **Test-time compute / inference-time scaling**: the broader, increasingly central 2025-2026 idea (behind reasoning models like OpenAI's o-series and DeepSeek-R1) that model quality/accuracy on hard problems can be improved not just by scaling up *training*-time compute (bigger model, more training data) but by **spending more compute at inference time** — e.g., generating much longer chain-of-thought traces, exploring/self-verifying multiple candidate solution paths, or iteratively refining an answer — before committing to a final output. This has become a distinct, complementary axis of scaling alongside traditional pretraining scaling laws (§31).

**Adding knowledge to an LLM — RAG vs. long-context vs. fine-tuning (a very common "when would you choose X" question):**
- **RAG (Retrieval-Augmented Generation)**: retrieve relevant documents/passages from an external knowledge source at query time and insert them into the prompt as context. Best for: knowledge that changes frequently/needs to stay current (no retraining needed to update), very large knowledge bases that can't fit in any context window, and needing **verifiable, cite-able sources** for generated claims. Downsides: retrieval quality is a hard dependency/bottleneck (garbage in, garbage out), added system complexity (a whole retrieval pipeline to build/maintain/evaluate), and extra latency per query.
- **Long-context**: simply put all relevant information directly in the prompt (feasible now that many models support 100K-1M+ token context windows). Best for: a bounded, known set of documents relevant to a session/task (e.g., "answer questions about this specific 200-page contract"), avoiding retrieval-pipeline complexity/failure modes entirely. Downsides: cost/latency scales with context length, and empirically, models can still struggle with reliably using information "buried" deep in a very long context ("lost in the middle" effects), so it's not a strictly superior free lunch over RAG even when context limits allow it.
- **Fine-tuning**: bake knowledge/behavior directly into the model's weights. Best for: teaching a stable **style, format, or skill/behavior** (not a good fit for injecting frequently-changing factual knowledge, since the model would need constant retraining) — knowledge injected purely via fine-tuning is also harder to verify/cite and can be forgotten or distorted through further training. In practice, many production systems **combine all three**: fine-tune for behavior/style/domain adaptation, use RAG for up-to-date/large-scale factual grounding, and use long-context for session-specific documents — this "it depends, and often it's a combination" framing is usually the strongest answer.

---

<a name="p4-12"></a>
### 37. Evaluating LLMs & RAG Systems
*Increasingly framed by interviewers in 2026 as "eval is the new system design" — a favorite closing/depth-testing topic.*

**Why classic ML metrics fail for open-ended generation:** metrics like accuracy or exact-match assume a single "correct" answer, but open-ended generation (chat, summarization, creative writing) often has **many valid answers**, varying in quality along dimensions (helpfulness, correctness, tone, conciseness) that a simple string-match can't capture. Even metrics like BLEU/ROUGE (n-gram overlap with a reference answer, from older NLP eval) correlate poorly with actual human judgments of quality for modern open-ended LLM outputs.

**Hallucination & calibration:** **hallucination** = the model generates fluent, confident-sounding text that is factually incorrect or unsupported/fabricated (not grounded in any real source or the provided context). **Calibration** = whether the model's expressed confidence (either explicitly stated, or implicitly via token probabilities) actually matches its true likelihood of being correct — a well-calibrated model that says "I'm 70% sure" should actually be right about 70% of the time it says that; LLMs (especially after heavy RLHF-style alignment) are often observed to be **overconfident/poorly calibrated**, stating incorrect facts with the same fluent confidence as correct ones, which is exactly what makes hallucination a hard, high-stakes practical problem (users can't easily tell confident-and-right from confident-and-wrong from tone alone).

**The RAG triad (RAGAS-style evaluation)** — three complementary metrics for RAG system quality, each isolating a different potential failure point in the pipeline:
- **Faithfulness (a.k.a. groundedness)**: does the generated answer's claims actually **follow from/are supported by** the retrieved context, or did the model hallucinate/add unsupported claims beyond what was retrieved? (measures the generator's fidelity to its given source material)
- **Answer relevance**: does the generated answer actually **address the user's question** (as opposed to being faithful to the retrieved context but off-topic/non-responsive to what was actually asked)?
- **Context relevance**: did the **retrieval step** actually fetch passages that are relevant/useful for answering the question in the first place (isolates whether failures are a retrieval problem vs. a generation problem)?
- Isolating these three lets you **diagnose where a RAG pipeline is failing** — e.g., low context relevance → fix your retriever/embeddings/chunking; high context relevance but low faithfulness → the generator is hallucinating despite having good source material; high faithfulness but low answer relevance → the model is accurately summarizing irrelevant retrieved content instead of addressing the actual question.

**Retrieval-specific metrics**: **recall@k** (of all truly relevant documents, what fraction appear in the top-k retrieved results?), **MRR (Mean Reciprocal Rank)** (averages `1/rank` of the first relevant result across queries — rewards getting a relevant result near the very top), **nDCG (normalized Discounted Cumulative Gain)** (accounts for *graded* relevance, i.e., some results are "somewhat relevant" and others "highly relevant," not just binary relevant/not, and discounts relevant results that appear lower in the ranked list).

**LLM-as-judge:** use a strong LLM to **evaluate/score** another model's outputs (e.g., rate response quality on a 1-10 scale, or given two responses, pick the better one) — scales far better than human evaluation and correlates reasonably well with human judgment in many settings, but has known biases worth naming if asked (**position bias** — favoring whichever response happens to appear first/second in the prompt; **verbosity bias** — favoring longer responses regardless of actual quality; **self-preference bias** — a judge model rating outputs from its own model family more favorably) — mitigations include randomizing response order, using multiple diverse judge models, and periodically validating judge scores against a smaller human-labeled sample.

**Pairwise win-rate / Arena-style evaluation (Elo):** rather than absolute scoring, have human or AI judges perform **pairwise comparisons** between two models' outputs on the same prompt ("which is better?"), then aggregate many such pairwise comparisons into an **Elo rating** (borrowed from chess ratings) for each model — pairwise judgments tend to be more reliable/consistent than asking for an absolute numeric quality score directly, since "which of these two is better" is a cognitively easier and more consistent judgment task than "rate this a 7 vs an 8."

**Golden sets & regression testing:** maintain a curated, fixed set of representative test prompts (a "golden set," ideally covering known edge cases/prior failure modes) with either fixed reference answers or fixed grading criteria, and **re-run it on every model/prompt/pipeline change** to catch regressions before shipping — the direct LLM-era analog of a software unit/regression test suite, essential for any production system since improving one behavior via prompt/fine-tuning changes can silently regress another.

**Agent-specific evaluation metrics** (for tool-using/agentic systems): **tool-selection quality** (did the agent choose the correct tool/API for the sub-task, out of the available options?), **task/step success rate** (did each intermediate step, and the overall task, complete correctly?), **trajectory adherence** (did the agent's sequence of actions follow a sensible/expected path, rather than looping, taking unnecessary detours, or diverging from the plan?) — these go beyond simply grading the final output, since an agent's *process* (efficiency, correct tool use, avoiding harmful/irreversible actions along the way) matters independently of whether it eventually stumbled into the right final answer.

**Benchmarks (know a few by name and roughly what they test):** **MMLU** (broad multi-subject academic/professional knowledge, multiple-choice, 57 subjects), **GPQA** ("Google-proof" graduate-level science QA, designed to be hard even to look up), **SWE-bench** (real-world GitHub software engineering issues — can the model actually generate a correct code patch that resolves the issue and passes tests?). **Safety/red-teaming & jailbreak robustness**: systematically probing a model with adversarial prompts designed to bypass its safety training (elicit disallowed content, leak system prompts, etc.) — an increasingly standard, explicit part of pre-deployment evaluation for any production LLM system, not an optional afterthought.

---

## Part V — Multimodal & Generative AI

<a name="p5-1"></a>
### 38. Multimodal Foundation Models

**Core idea:** build a model with a **shared representation space** across different modalities (text, image, audio, video, and even robot actions) so that the model can reason across, translate between, and jointly generate multiple modalities — rather than needing entirely separate, siloed models per modality.

**Fusion approaches (know the spectrum, from loosest to tightest coupling):**
- **Contrastive dual-encoder (CLIP / SigLIP)**: train **two separate encoders** (one for images, one for text) such that matched image-text pairs end up with **high cosine similarity** between their embeddings, and mismatched pairs end up with low similarity — trained via a contrastive loss over large batches of (image, caption) pairs scraped from the web (for each image, its true caption should score higher than all other captions in the batch, and vice versa). This produces a shared embedding space useful for retrieval, zero-shot classification (compare an image's embedding to embeddings of candidate text class labels), but the two encoders remain **architecturally separate** — there's no deep fusion of the two modalities into one joint reasoning stack. **SigLIP** improves on CLIP's contrastive loss by replacing the softmax-based contrastive loss (which requires comparing against *all* other examples in the batch, and needs careful batch-size/normalization handling) with a simpler **sigmoid loss** applied independently to each image-text pair (binary "does this pair match, yes/no"), which scales better and doesn't require the same global batch normalization.
- **Projection / adapter into LLM token space (LLaVA)**: take a frozen (or lightly fine-tuned) pretrained vision encoder's output, pass it through a small trainable **projection layer** (e.g., a linear layer or small MLP) that maps image features into the **same embedding dimensionality/space as the LLM's token embeddings**, then feed those "image tokens" directly into the LLM alongside text tokens, letting the LLM's existing self-attention naturally mix visual and textual information. This is architecturally simple (reuses a powerful pretrained LLM almost unchanged) and is the dominant recipe behind most open VLMs — see §39.
- **Cross-attention (Flamingo-style)**: rather than injecting image features as tokens directly into the input sequence, insert **dedicated cross-attention layers** interleaved within the LLM's existing layers, where text tokens attend to visual features via a separate cross-attention mechanism (Q from text, K/V from vision features) — keeps the LLM's own self-attention computation focused purely on text while still allowing rich visual grounding, and can be more efficient for handling many/high-resolution images without bloating the main token sequence length.
- **Early vs. late fusion**: *early fusion* combines modalities right at the input (e.g., tokenizing all modalities into a shared token space early on, as in native multimodal training); *late fusion* keeps modalities mostly separate through most of the network and only combines them near the output. Early fusion generally allows richer cross-modal interaction throughout the network's depth but is more complex/expensive to train from scratch.
- **Native/"omni" any-to-any models (GPT-4o, Gemini)**: trained from the ground up to natively handle **arbitrary combinations of input and output modalities** within a single unified model/architecture (text, image, audio in; text, image, audio out) — rather than bolting a vision encoder onto a pretrained text-only LLM after the fact, these models are designed and trained jointly across modalities from the start, which (per reported results) tends to yield better cross-modal reasoning and lower latency (e.g., for real-time audio conversation) than pipelines stitching together separate unimodal models.

**Unified understanding AND generation; tokenizers for generation:** for a model to *generate* images/audio (not just understand/describe them), continuous modalities need to be **discretized into tokens** first (so the same next-token-prediction / autoregressive machinery used for text can apply) — **VQ-VAE (Vector-Quantized Variational Autoencoder)** is the classic technique: an encoder compresses an image into a grid of continuous latent vectors, each of which is then **snapped to the nearest entry in a learned, finite "codebook"** of discrete embedding vectors (this quantization step is what makes the representation discrete/token-like), and a decoder reconstructs the image from these discrete codebook indices — giving you a vocabulary of "image tokens" exactly analogous to a text vocabulary, which an autoregressive transformer can then be trained to predict, enabling unified AR generation across text and image tokens in one model.

---

<a name="p5-2"></a>
### 39. Vision-Language Models (VLMs)

**Standard architecture (the pattern to draw on a whiteboard):**
```
Image → [Vision Encoder: ViT / SigLIP / DINOv2] → [Projector: linear/MLP] → [LLM] → text output
```
- **Vision encoder**: typically a **ViT (Vision Transformer)**-based model, pretrained either via supervised classification, contrastive learning (CLIP/SigLIP-style), or self-supervised methods (**DINOv2** — trained with a self-distillation objective without any labels or captions at all, producing strong general-purpose visual features praised for capturing fine-grained spatial/semantic structure well, particularly useful for tasks needing precise localization). This encoder converts an image into a set of patch-level feature vectors (an image is split into fixed-size patches, e.g., 14×14 or 16×16 pixels, each treated like a "token").
- **Projector**: a small trainable module (often just a linear layer, or a 2-layer MLP) that maps the vision encoder's output dimensionality into the LLM's embedding dimensionality, so visual features can be "spoken in the same language" as the LLM's text token embeddings.
- **LLM backbone**: a pretrained (usually frozen initially, then often unfrozen/fine-tuned in later training stages) language model that receives the projected visual tokens interleaved with text tokens and processes them together via its normal self-attention mechanism.

**Training recipe (two-stage, the standard/LLaVA-style pattern):**
1. **Image-text pretraining / feature alignment**: train (usually just) the lightweight projector on large-scale, relatively noisy image-caption pairs, with the vision encoder and LLM both **frozen** — this cheaply teaches the projector to map visual features into a space the (already-capable) LLM can meaningfully interpret, without touching/risking the expensive pretrained encoders.
2. **Visual instruction tuning**: fine-tune (the projector, and typically the LLM too, sometimes the vision encoder as well) on a smaller, higher-quality dataset of **(image, instruction, response)** triples — teaching the model to actually follow visual instructions/questions in a helpful, conversational way, directly analogous to how text-only SFT (§32) teaches instruction-following on top of raw pretraining.

**Key VLM tasks (know what each means concretely):**
- **VQA (Visual Question Answering)**: answer a natural-language question about an image's content.
- **Captioning**: generate a natural-language description of an image.
- **OCR / document understanding**: read and interpret text embedded within images (scanned documents, screenshots, receipts) — often requires specialized handling of high-resolution image input (since text can be tiny relative to a full image) and layout-awareness (understanding tables, forms, multi-column layouts, not just raw text).
- **Grounding / detection**: localize *where* in the image a described object/region is (e.g., output bounding-box coordinates for "the red car" mentioned in text) — connects language references to specific spatial regions of the image.
- **Chart / UI understanding**: interpreting structured visual information like charts, graphs, and application/website screenshots (an increasingly important capability for agentic use cases — e.g., a model that can "see" and act on a UI, relevant to computer-use agents).

**Notable models to be able to name-drop with a one-line differentiator:** **LLaVA** (the original, simple/popular open-source linear-projector recipe described above), **PaliGemma** (pairs a SigLIP vision encoder with the Gemma LLM, notable for strong performance at a relatively small/efficient scale, widely used as a VLM base for further fine-tuning including in robotics, e.g., π0 in §40), **InternVL** (known for scaling the vision encoder itself to very large sizes, arguing vision-encoder scale matters as much as LLM scale for VLM quality), **Qwen-VL** (strong open-weight VLM family with notable OCR/document and grounding capabilities), and the frontier closed models **GPT-4o**, **Gemini**, and **Claude** (which handle vision natively as part of a broader, often now-multimodal-from-the-ground-up, architecture rather than a bolted-on VLM adapter in the more recent generations — see §38's "native/omni" discussion).

---

<a name="p5-3"></a>
### 40. Vision-Language-Action Models (VLAs)

**Core idea:** extend the VLM recipe (§39) to **embodied/robotic control** — instead of (or in addition to) outputting text, the model outputs **robot actions**, closing the perceive → understand → act loop, often within a **single forward pass** of one unified model (rather than a traditional robotics pipeline with separate perception, planning, and control modules built and tuned independently).

**Why build on VLMs specifically (the key motivating insight)?** A model pretrained on massive internet-scale image-text data has already learned rich, general visual and semantic world knowledge (what objects look like, what words like "left of," "stack," "fragile" mean) — VLAs aim to **transfer that broad web-scale knowledge and even chain-of-thought-style reasoning to robot control**, rather than training a robot control policy from scratch on the comparatively tiny/expensive amount of real-world robot interaction data that's available. This is a major reason VLAs have shown notably better generalization (to novel objects, instructions, and scenes) than earlier, narrowly-trained robot learning approaches.

**Action representation — two competing approaches (a good cross-question to be ready for):**
- **Discrete action tokens** (RT-2, OpenVLA): represent continuous robot actions (e.g., end-effector position/rotation deltas, gripper open/close) by **discretizing/binning each action dimension into a fixed number of bins**, mapping each bin to a token ID — this lets you literally reuse the exact same autoregressive next-token-prediction machinery and vocabulary mechanism as language, treating "predict the next action" as structurally identical to "predict the next word." Simple, elegant, directly reuses existing LLM/VLM training infrastructure — but the discretization introduces a hard resolution/precision ceiling on action granularity, and generating actions one discrete token at a time autoregressively can be slow for high control frequencies.
- **Continuous action heads via diffusion / flow-matching** (π0): instead of discretizing actions into tokens, attach a **separate specialized action-generation head** (using diffusion or, in π0's case, **flow-matching** — a related but generally faster-to-sample generative technique, see §41) on top of the VLM backbone, which directly outputs continuous, high-precision action values, and critically, can generate them at **much higher frequency** (π0 targets ~50Hz control) than token-by-token autoregressive decoding would practically allow — important for smooth, precise, dexterous manipulation tasks where discretization artifacts or generation latency would be problematic.

**Notable models (know these specifics — they're explicitly called out as high-value facts):**
- **RT-2** (Google DeepMind): built on top of **PaLI-X / PaLM-E** vision-language models; its key contribution was demonstrating that transferring web-scale VLM knowledge (including some emergent chain-of-thought-style reasoning about a scene before acting) directly improves real-world robot generalization to novel objects/instructions, compared to robot policies trained without that VLM foundation.
- **OpenVLA**: a fully **open-source**, **7B**-parameter VLA, combining **DINOv2 + SigLIP** vision features with a **Llama-2** language backbone, trained on **970K real robot demonstration episodes**. The headline result worth memorizing: it **outperforms RT-2-X (a 55B-parameter model) despite being ~7× smaller** in parameter count — a strong, quotable data point for "does bigger always mean better" discussions, suggesting data quality/diversity and architecture choices can matter as much as raw scale for this task.
- **π0 (Pi-Zero)**: built on the **PaliGemma** VLM (§39), with a **flow-matching action expert** module attached for continuous, high-frequency (~50Hz) dexterous action generation — positioned as a "generalist" robot policy aimed at broad applicability across many robot embodiments/tasks rather than a single narrow task.

**Data sources for VLA training**: **teleoperation demonstrations** (a human directly/remotely operates the robot to perform a task, recording the resulting state-action trajectories as training data) and **Open X-Embodiment** (a large, standardized, multi-institution collaborative dataset aggregating robot demonstration data across many different robot types/embodiments and tasks — analogous in spirit to how ImageNet or Common Crawl served as foundational shared datasets for vision/language, aiming to provide the "internet-scale" (relatively speaking) data needed for broadly generalist robot policies).

**Use cases**: robot manipulation (pick-and-place, tool use), humanoid robot control, and the broader goal of **generalist robot policies** — a single model capable of performing many different tasks across different environments/robot embodiments, rather than one narrowly-trained model per specific task.

---

<a name="p5-4"></a>
### 41. Diffusion vs. Autoregressive Generation

| | **Autoregressive (AR)** | **Diffusion** |
|---|---|---|
| How | predict next token sequentially | iterative denoising from noise |
| Likelihood | exact | variational / score-based |
| Strength | discrete sequences, variable length, reasoning | continuous high-dim (image/video/audio), high fidelity |
| Speed | 1 forward pass / token (KV-cache helps) | many denoising steps (cut via distillation/consistency/flow matching) |
| Examples | text (GPT), image tokens (Parti, VAR), audio (AudioLM) | image (Stable Diffusion, Imagen), video (Sora, Veo — DiT), audio/music (Stable Audio), robot actions (π0) |

**Autoregressive (AR) generation — the deep dive:** discretize the target modality into a sequence of tokens (for text: subword tokens via BPE, §31; for images/audio: via a learned tokenizer like VQ-VAE, §38), then train a transformer to predict each token conditioned on all previous tokens via **exact** next-token-prediction likelihood (the joint probability of the full sequence factorizes exactly as a product of conditionals via the chain rule of probability — no approximation needed, unlike diffusion's variational bound). This unified, token-based approach is what enables **unified multimodal models** — since text, image, and audio can all be represented as sequences of discrete tokens from (possibly different, possibly shared) vocabularies, the same autoregressive transformer architecture and training objective can, in principle, handle all of them within one model. **Image-specific AR**: rather than raster-scanning pixel-by-pixel or patch-by-patch (which doesn't respect an image's inherent 2D/multi-scale structure well), techniques like **VAR (Visual AutoRegressive modeling)** instead generate an image via **next-scale prediction** — predicting a coarse, low-resolution version of the image first, then progressively finer/higher-resolution refinements, which better matches images' natural coarse-to-fine structure than a naive raster-order token sequence. **Audio AR**: models like **AudioLM/MusicGen** apply the same discretize-then-autoregress recipe to audio, using learned audio tokenizers (often based on residual vector quantization) to convert continuous waveforms into discrete token sequences.

**Diffusion generation — the deep dive:**
- **Forward process**: gradually add small amounts of Gaussian noise to a real data sample (e.g., an image) over many steps, until it becomes pure random noise — this process is fixed/not learned, just a mathematical noising schedule.
- **Reverse process (what's actually learned)**: train a neural network to **predict and remove** the noise added at each step, progressively — i.e., given a noisy image at noise-level `t`, predict either the noise that was added (**noise prediction**, the DDPM formulation) or a **score function** (the gradient of the log-probability density, closely related — score-based and noise-prediction formulations are mathematically connected/near-equivalent framings of the same underlying idea). Starting from pure random noise at inference time, repeatedly applying this learned denoising step gradually "sculpts" the noise into a realistic sample.
- **Latent diffusion (Stable Diffusion)**: rather than running the (computationally expensive) diffusion process directly in high-resolution pixel space, first compress the image into a much smaller **latent space** using a pretrained autoencoder (a VAE), run the entire diffusion (forward and reverse) process in that smaller, more compact latent space, then decode the final denoised latent back into a full-resolution pixel image at the very end — a major efficiency win (diffusion in a small latent space is far cheaper than in full pixel space) that's what made high-resolution diffusion image generation computationally practical/accessible.
- **DiT (Diffusion Transformers)**: replaces the U-Net (convolutional) architecture originally used as the denoising network in most diffusion models with a **transformer** architecture instead — transformers scale more predictably/favorably with more compute and data (benefiting from the same scaling-law insights, §31, that drove transformer adoption in language) and are the backbone architecture behind state-of-the-art **video** generation models like **Sora** and **Veo**, which additionally need to model temporal (across-frame) consistency/coherence on top of per-frame spatial structure.
- **Classifier-free guidance (CFG)**: a widely-used inference-time technique to improve how closely generated samples follow a given conditioning signal (e.g., a text prompt) — the model is trained to (randomly, some fraction of the time during training) also handle an **unconditional** case (no text conditioning), and then at inference time, the final prediction is computed as an **extrapolation away from the unconditional prediction, toward the conditional (text-guided) prediction**: `guided_pred = uncond_pred + w * (cond_pred - uncond_pred)`, where `w > 1` (the "guidance scale") pushes generation to adhere to the prompt more strongly than the model's raw conditional prediction alone would — a higher guidance scale generally improves apparent prompt-adherence but can reduce output diversity/naturalness if pushed too far.
- **Fast samplers (DDIM)**: standard diffusion sampling (DDPM) requires many (e.g., 1000) small denoising steps for high quality, which is slow. **DDIM (Denoising Diffusion Implicit Models)** reformulates the sampling process to be **non-Markovian** (each step doesn't have to strictly depend only on the immediately preceding step in the same way), allowing you to **skip steps** and generate high-quality samples in far fewer steps (e.g., 20-50) with minimal quality loss, without needing to retrain the underlying model — a pure sampling-algorithm improvement.
- **Flow matching / rectified flow (SD3, π0)**: a more recent, related generative framework that, instead of the (stochastic, multi-step noise-adding/removing) diffusion formulation, directly learns a **deterministic vector field ("flow")** that transports samples from a simple noise distribution to the target data distribution along (ideally) straight-line paths in a mathematically cleaner, more direct way. Because the learned transport paths tend to be straighter/simpler than a typical diffusion trajectory, flow matching often needs **fewer sampling steps** for comparable quality (faster generation), which is exactly why it's been adopted both in cutting-edge image generation (Stable Diffusion 3) and in the continuous robot-action generation head of **π0** (§40), where fast, high-frequency generation is a hard practical requirement.

**Emerging: unifying AR and diffusion:** an active 2025-2026 research direction includes **text diffusion / masked diffusion language models** (applying diffusion-style iterative refinement to *discrete* text generation, as an alternative to strictly left-to-right autoregressive generation — potentially allowing more parallel, non-sequential generation and easier "fill in the blank"/editing-style generation), **consistency models & few-step models** (distilling a slow, many-step diffusion model down into a model that can generate high-quality samples in just 1-4 steps, trading a bit of quality for massive speed gains), and **unified AR + diffusion stacks** (hybrid architectures that use autoregressive generation for some aspects/modalities — e.g., high-level structure or discrete tokens — combined with diffusion for others — e.g., fine-grained continuous detail), reflecting a broader trend of the field converging toward flexible, general-purpose architectures rather than strictly siloed "AR for text, diffusion for images" camps.

---

## Part VI — Rapid-Fire Cheat Sheet
*Use this as a final self-test the night before / morning of. If you can answer each in 1-3 sentences without looking, you're ready. (§ = section to review if you can't.)*

**Classic ML**
- ML vs traditional programming? → learn the rule from data instead of hand-coding it. §1
- Supervised vs unsupervised vs semi-supervised? → labeled / unlabeled / mix. §1
- Bias-variance trade-off? → error = bias² + variance + irreducible; simple models=high bias/underfit, complex=high variance/overfit. §14
- Overfitting prevention checklist? → more data, regularization, simpler model, ensembling, CV, augmentation, feature selection. §14
- Why train/val/test split? → tune on val, report once on untouched test to avoid leakage. §10
- Cross-validation, good k? → k-fold averages k train/test splits; k=5 or 10 is the standard trade-off. §10
- L1 vs L2 regularization? → L1/Lasso → sparsity (some weights exactly 0); L2/Ridge → smooth shrinkage, handles multicollinearity. §14
- Missing/corrupted data handling? → diagnose MCAR/MAR/MNAR, then delete/impute (mean/median/model-based/MICE)/flag-missing, or use a model that handles it natively (XGBoost). §16
- Decision tree — how does it work? → recursive greedy splits by impurity reduction (Gini/entropy), leaves store predictions, prune to prevent overfitting. §4
- Logistic regression? → linear score → sigmoid → probability, trained via cross-entropy (convex, MLE-derived). §2
- KNN? → supervised, lazy/non-parametric, classify by majority vote of k nearest labeled neighbors. §11 (cross-question box)
- K-means vs KNN? → K-means = unsupervised clustering; KNN = supervised classification. Don't confuse them. §11
- Random forest / GBDT? → RF = bagging + random feature subsets (decorrelates trees, reduces variance); GBDT = sequential trees fit to residuals/gradients (reduces bias). §5
- Gradient descent? → iteratively step opposite the gradient to minimize loss; SGD/mini-batch/momentum/RMSprop/Adam are all refinements. §6
- SVM / kernel SVM? → max-margin separator; kernel trick computes dot products in a higher-dim space without explicit mapping (RBF is default nonlinear kernel). §3
- Neural networks — how do they work? → layers of weighted sums + nonlinearities, trained via backprop (chain rule) + gradient descent. §17
- Deep learning vs traditional ML? → deep learning uses many-layer neural nets that learn features automatically from raw data; traditional ML often needs hand-engineered features and simpler models. §17, §1
- Backpropagation? → chain rule applied backward through the computational graph to get every parameter's gradient efficiently. §17
- CNN, how does it work? → convolutional filters slide over input sharing weights, detecting local spatial patterns (edges→shapes→objects) with far fewer parameters than a fully-connected net; pooling downsamples for translation invariance. (Not detailed above — know this as background.)
- Transfer learning? → reuse a model pretrained on a large/general dataset (freeze/fine-tune some layers) for a new, often smaller/related task — saves data and compute vs. training from scratch.

**LLMs / GenAI**
- Why divide attention by √dₖ? → keeps QKᵀ variance ~1, avoids softmax saturation/vanishing gradients. §26 (in §21's deep dive)
- KV cache, why, memory scaling? → cache K/V once per token instead of recomputing; scales with layers×heads×head_dim×seq_len×batch. §28
- MHA vs MQA vs GQA? → MHA=full K/V per head (best quality, biggest cache); MQA=1 shared K/V (fastest, quality drop); GQA=grouped sharing, Llama's chosen sweet spot. §26
- RoPE, why preferred? → rotates Q/K so dot product depends on relative position; extrapolates to longer context better than learned absolute embeddings. §27
- RAG vs fine-tuning vs long-context? → RAG for current/large/citable knowledge; long-context for a bounded known doc set; fine-tuning for stable behavior/style, not fast-changing facts. §36
- SFT vs DPO vs GRPO vs RLVR? → SFT=demos; DPO=preference pairs, no RL loop; GRPO=RL without a critic, group-relative advantage; RLVR=verifiable/rule-based rewards (math/code). §33
- Why does GRPO drop the critic? → estimates advantage from a sampled group's relative rewards instead of a learned value network — cheaper, still variance-reduced. §33
- LoRA / QLoRA, when to use PEFT? → freeze base weights, train small low-rank ΔW=BA; QLoRA additionally quantizes the frozen base to 4-bit. Use PEFT over full FT when compute/memory-constrained or want to avoid catastrophic forgetting/store many cheap task adapters. §34
- Quantization, accuracy/latency tradeoff? → lower bit-width = smaller/faster but riskier accuracy; PTQ (GPTQ/AWQ) is cheap/fast to apply, QAT is more accurate at very low bits but needs retraining. §35
- Speculative decoding? → small draft model proposes tokens, big model verifies in parallel, accept matches — exact output, faster wall-clock. §35
- LLM/RAG evaluation, RAG triad? → faithfulness (grounded in context?), answer relevance (answers the question?), context relevance (did retrieval fetch the right stuff?). Plus LLM-as-judge, benchmarks (MMLU/GPQA/SWE-bench). §37
- VLM architecture? → vision encoder (ViT/SigLIP/DINOv2) → projector → LLM, trained via feature-alignment pretraining then visual instruction tuning. §39
- VLA, discrete vs continuous actions? → RT-2/OpenVLA discretize actions into tokens (reuse LLM machinery); π0 uses a flow-matching head for continuous, high-frequency (~50Hz) control. §40
- Diffusion vs AR? → AR predicts tokens sequentially with exact likelihood, great for discrete/reasoning; diffusion iteratively denoises, great for high-fidelity continuous data (image/video/audio), faster via distillation/flow-matching. §41
- Classifier-free guidance? → extrapolate the conditional prediction away from the unconditional one to increase prompt adherence. §41
- Flow matching / rectified flow? → learns a deterministic, near-straight-line transport from noise to data — fewer steps needed than diffusion, used in SD3 and π0. §41
- Mixture-of-Experts, active vs total params? → route each token to top-k of many experts via a learned router; total params = full model size, active params = what's actually computed per token (much smaller) — needs a load-balancing loss/bias to avoid expert collapse. §30

---

## Part VII — Further Resources

**Foundational courses**
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) ([YouTube lectures](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN))
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) or [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

**Quick review / refreshers**
- [StatQuest Machine Learning playlist](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — best for building/re-building intuition fast, very high yield per minute watched.
- [StatQuest Statistics playlist](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) — for p-values/R²/statistical concepts.
- [ML Cheatsheets (readthedocs)](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML flashcards](https://machinelearningflashcards.com/)
- [45 ML interview questions (Simplilearn)](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions)

**Transformers / LLMs**
- [The Illustrated Transformer (Jay Alammar)](http://jalammar.github.io/illustrated-transformer/) — read this immediately before any interview where "explain the transformer" might come up; it's the resource every other explanation of transformers (including this doc's §21) is standing on.
- For end-to-end GenAI **system design** (RAG pipelines, agents, serving infrastructure) — this doc covers component-level LLM concepts only; pair it with system-design-focused material, e.g., the referenced [Agentic AI Systems repo](https://github.com/alirezadir/Agentic-AI-Systems.git).

**How to actually use this doc in the time you have**
1. **First pass (fast):** read Part VI (Rapid-Fire Cheat Sheet) top to bottom — this tells you immediately which topics you already have cold vs. which need work.
2. **Second pass (targeted):** for every rapid-fire item you hesitated on, jump to its linked section number and read the full explanation + cross-questions.
3. **Third pass (say it out loud):** pick 5-10 sections at random and explain them out loud, unscripted, in under 2 minutes each, as if to an interviewer — this is the single highest-value prep activity, since interviews are verbal, not silent-reading exams.
4. **Day-of:** skim Part VI once more right before you go in.

---
*End of guide. Good luck — you've now got the full breadth this document's own table of contents demands, from least-squares residuals to flow-matching robot policies.*
