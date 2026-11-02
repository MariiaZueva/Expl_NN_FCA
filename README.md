# Project "Conceptual Views for Global Interpretation of Neural Networks"
A big research-oriented homework for the **Ordered Sets for Data Analysis** course.

This project merges **neural networks** and **Formal Concept Analysis (FCA)** — but in the opposite direction from Neural FCA. Rather than building an interpretable network from a lattice, we take an arbitrary pretrained network and analyze it **post-hoc, globally**: activations of the last hidden layer and weights of the output layer are turned into two FCA contexts, from which lattices, symbolic rules, and architecture-similarity measures are derived.

Pay attention on helpful resources with code provided:
- Libraries with FCA methods implemented:

  https://github.com/EgorDudyrev/FCApy
  
  https://github.com/smartFCA/caspailleur
  
- Base paper: Hirth, J., Hanika, T. *Conceptual Views of Neural Networks: A Framework for Neuro-Symbolic Analysis* (2025).

The main research question:

```text
Can a neural network be explained globally, post-hoc, and without retraining
by representing its last-hidden-layer activations and output-layer weights
as a pair of FCA contexts?
```

## Task Description

You need to choose a classification dataset, train several neural networks with different last-hidden-layer activations, extract the two views, scale them into formal contexts, build lattices and rules, and compare against local-explanation baselines. The core pipeline follows Hirth & Hanika (2025).

Minimum pipeline:

1. Choose an open classification dataset (tabular or image; see compute tiers below).
2. Train 3–5 neural networks with different last-hidden-layer activations (ReLU, Swish, Tanh); fix the seed, save checkpoints.
3. Extract the **many-valued conceptual view** `V = (O, W)`:
   - `O ∈ R^{|G|×|N|}` — Object View (activations);
   - `W ∈ R^{|C|×|N|}` — Class View (output weights). 
4. Apply **dichotomic scaling** with thresholds `δ_O, δ_W` to obtain the **symbolic conceptual view** `V_D = (O_D, W_D)` — a pair of formal contexts.
5. Build the concept lattice (or a constrained fragment) on the symbolic view; extract propositional implications.
6. Apply subgroup discovery to relate neurons to human-readable features (color, shape, taxonomy).

Main improvements expected beyond the baseline:

- threshold study (`δ = 0` vs mean / median / etc) crossed with activation;
- pure vs α-weak concept selection;
- your own scaling or similarity variant.

## Learning Goals

After completing the project, you should be able to:

1. Extract and interpret the many-valued conceptual view `(O, W)` of a trained network.
2. Apply dichotomic scaling to obtain a symbolic view and build a formal context.
3. Build a concept lattice (or a constrained fragment) and extract propositional implications.
4. Use subgroup discovery to derive human-comprehensible rules about neurons.

## Compute Budget

The paper's ImageNet setup (100k test images × 24 models × 1024–4096 neurons) is not feasible on a laptop. Use more compact dataset. For example:

| Dataset                                                 | Training              | Storage per model | Runtime (1 GPU/CPU) |
| ------------------------------------------------------- | --------------------- | ----------------- | ------------------- |
| Breast Cancer Wisconsin (tabular, 569 rows) + small MLP | from scratch, minutes | < 1 MB            | minutes             |

## To-do List

### Basic solution

- [ ] Choose an open classification dataset; define the target attribute.
- [ ] Train 3–5 networks with ReLU / Swish / Tanh in the last hidden layer; fix seeds; save checkpoints.
- [ ] Extract `O` and `W`.
- [ ] Apply dichotomic scaling (`δ = 0` for Tanh; ReLU as contrast); obtain `O_D, W_D`.
- [ ] Report class separation and SV-Fid (1-NN and decision tree surrogates).
- [ ] Build a constrained lattice fragment (min support, max intent size); extract implications.
- [ ] Apply subgroup discovery with human-readable features.
- [ ] Prepare a PDF report and a short presentation.

## Submission

Your repository should be standalone and follow the general course structure. It should contain:

- working code;
- one main notebook demonstrating the full pipeline (EDA → black box → views → scaling → lattice → rules → comparison);
- a PDF report;
- a short presentation;
- saved examples of lattice fragments, implications, and rules;
- `README.md` with reproduction instructions.

The report should describe model accuracy, fidelity tables, class separation, lattice fragments, and the comparison with local-explanation baselines.

**Tooling.** Use FCApy (caspallieur) for lattice construction and implication mining on the **symbolic** view only (i.e., after dichotomic scaling). The many-valued view `O, W` is handled with NumPy/pandas and is not an FCA object until it has been scaled.

###  Symbolic Conceptual View

For a network with last-hidden-layer neurons `N`, classes `C`, and objects `G`:

- **Object View** `O ∈ R^{|G|×|N|}`, `O[i,j] = n_j(g_i)`;
- **Class View** `W ∈ R^{|C|×|N|}`, `W[i,j] = w_{i,j}`.

Both are needed: `O` captures "what the object looks like to the network", `W` captures "what each class expects from the network". Reporting only `O` (as activation-based methods do) loses the class side.

The output for class `c_i` can be rewritten as

O(g) · W(c_i) + b = |O(g)| · |W(c_i)| · cos(O(g), W(c_i)) + b

so objects and classes live in the same space. This induces a pseudometric space `(G ∪ C, d̂_V)`; 1-NN on it is a faithful surrogate for the network.


Apply dichotomic scaling with thresholds `δ_O, δ_W`:

```text
O_D = (G, N ∪ N̄, I_O),  (g, n_j) ∈ I_O ⇔ n_j(g) > δ_O,  (g, n̄_j) ∈ I_O ⇔ n_j(g) ≤ δ_O
W_D = (C, N ∪ N̄, I_W),  (c_i, n_j) ∈ I_W ⇔ w_{i,j} > δ_W,  (c_i, n̄_j) ∈ I_W ⇔ w_{i,j} ≤ δ_W
```

Suitable thresholds depend on the architecture. For ReLU, the positive co-domain makes negated attributes `N̄` uninformative — this is why the paper recommends **Tanh** for the symbolic view. For Tanh, `δ = 0` gives an almost equal positive/negative split.

**Quality criteria for the symbolic view:**

1. **Class separation** — each class uniquely represented in `W_D`;
2. **SV-Fid** — fidelity of interpretable classifiers (1-NN, decision tree) trained on the symbolic context.

## Lattice, Rules, and Subgroup Discovery

Each algebraic relation corresponds to exactly one natural order via Galois connections. FCA reveals this order through formal concepts `(A, B)` with `A^I = B` and `B^I = A`; the set of all concepts forms a complete lattice `B(K)`, visualizable as a line diagram. Applied to networks, the lattice visualizes the conceptual hierarchy learned by the network, and its size is an upper bound on the number of learned concepts.

**Constraints to keep the lattice readable** (the paper reports up to 3.8M concepts at `|N| = 32`):

- min support;
- max intent size;
- top-n by interestingness (Kuznetsov & Makhalova);
- min purity.

**Subgroup discovery.** Build a formal context `C = (C, S_M, I_C)` of background knowledge (`S_M`: shapes, colors, taxonomy). Define a symbolic interpretation `S = (N, S_M, R)` with `(n, S_m) ∈ R ⇔ {n}^{I_w} ∼ {S_m}^{I_c}`, where `∼` is reflexive and symmetric (not necessarily transitive), inferred via subgroup discovery (`pysubgroup`). This expresses neurons in terms of human-readable features and vice versa:

> "Fruits that are not brown, not stained, not orange, and not star-shaped activate neuron 13."
> "If neurons 13, 14, and 9 are activated, the fruit is orange with confidence ≈ 0.54."

## Metrics

**Mandatory:**

- **Fidelity** (weighted and unweighted) — agreement of the surrogate with the black box, **not** accuracy vs ground truth.
- **Class separation** — uniqueness of classes in the symbolic view.
- **Compactness** — intent size, number of concepts, fragment size.
- **Purity** — for `C = (A, B)`: `purity_c(C) = |{z ∈ A : f(z) = c}| / |A|`.
- **Coverage** — fraction of objects explained by the selected concept set.
- **Stability** — see three-way split below.

**Stability (three axes).** "Different seeds" is not enough, because the symbolic view is deterministic given `(model, δ)`:

1. **Training stability:** retrain with different seeds → compare views.
2. **Threshold stability:** vary `δ` around the chosen value → Jaccard on concept intents and ΔFidelity.
3. **Subsample stability:** bootstrap `G` → compare lattice fragments (Jaccard on intents).

## Experimental Protocol

Minimum protocol:

- one dataset;
- stratified train/test split with fixed `random_state`;
- primary metric: **fidelity** to the black box (plus class separation);
- at least 4 standard surrogates (kNN, LogReg, SVM, Decision Tree, Random Forest, XGBoost);
- at least one Conceptual Views configuration.

Fixed parameters to report:

**Black box**
- architecture, `|N|`, activation, seed;
- model accuracy on the test set.

**Symbolic view**
- thresholds `δ_O, δ_W`;
- class separation table;
- SV-Fid table (activation × `δ`);

**Lattice**
- min support, max intent size, top-n;
- lattice size;
- lattice construction algorithm;
- number of extracted implications.

**Rules**
- subgroup discovery settings;
- support and confidence of derived rules.

## Interpretability Analysis

Interpretability is evaluated separately from fidelity. The report must include:

1. **Semantic interpretation of neurons.** Pick 3–5 concept neurons with the largest absolute weights to the output layer; explain intent `B`, extent `A`, and purity w.r.t. the target class.
2. **Interpretation of weights.** Show concept→class weights and explain their conditional-probability meaning (e.g., "concept C implies class + with probability 0.87").
3. **Activation path for a specific object.** For 1–2 test objects, trace the activation from input attributes through concepts to the output class and write a natural-language explanation.
4. **Effect of concept selection.** Compare different `δ` and different activation functions; check whether smaller, purer concept sets keep meanings clean.
5. **Network visualization.** Provide a lattice fragment (nodes = concepts with intents, edges = covering relation with weights).

## Typical Mistakes

- Evaluating surrogate accuracy (vs ground truth) instead of fidelity (vs the black box). The paper is explicit that fidelity, not accuracy, is the right criterion.
- Ignoring the choice of `δ`. The paper shows `δ = 0` works for Tanh, but does **not** establish it for Swish or ReLU. Treat `δ` as a hyperparameter to be studied.
- Building a global lattice without constraints — it becomes unreadable. Constrain by min support or max intent size.

## Check Questions

1. How does the object view differ from the class view, and why are both needed?
2. Why is Tanh recommended for the symbolic view rather than ReLU?
3. How are propositional implications extracted from the lattice?

## Recommended References

1. Hirth, J., Hanika, T. *Conceptual Views of Neural Networks: A Framework for Neuro-Symbolic Analysis* (2025).
2. Ganter, B., Wille, R. *Formal Concept Analysis: Mathematical Foundations.* Springer.
3. Kuznetsov, S. O., Makhazhanov, N., Ushakov, M. *On neural network architecture based on concept lattices* (2017).
4. Kuznetsov, S. O., Makhalova, T. *On interestingness measures of formal concepts* (2018).
