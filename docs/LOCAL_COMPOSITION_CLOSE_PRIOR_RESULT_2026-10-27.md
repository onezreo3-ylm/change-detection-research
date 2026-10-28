# Local action composition: closest prior check

2026-10-27. This is a bounded primary-source review, not novelty certification.
The candidate remains actual action-effect prediction and constrained joint
choice on the original change detector. Its supporting normalization, caching,
local masks and generic decision rules must be credited as existing components.

| Primary source and inspected scope | Overlap and consequence for this candidate |
| --- | --- |
| [SIGE v3, section3.2](https://arxiv.org/html/2211.02048v3), normalization paragraph; [author project](https://www.cs.cmu.edu/~sige/) | SIGE explicitly reuses original activation statistics for instance/group normalization, replacing normalization by scale and shift. It caches original activations and recomputes edited blocks with overlapping convolution halos, with specialized kernels to turn sparsity into measured latency reductions. Our anchored replay implements the same borrowed-statistics affine principle; preserving unchanged activations and propagating local supports are not new algorithmic contributions. The present dense full-decoder wrapper has no sparse-kernel speed claim. Exact output/count composition for disjoint supports is a property to validate for this fixed graph, rather than evidence that caching was invented here. |
| [FISEdit, AAAI2024 author abstract](https://ojs.aaai.org/index.php/AAAI/article/view/29599) | Reuses cached feature maps/statistics outside affected editing regions and identifies changed regions. The abstract suffices to exclude a claim that automatic region selection plus cache reuse is new. Its complete algorithm was not inspected in this bounded pass; do not assert a stronger equivalence or absence of action prediction. |
| [SparseRefine, author method description](https://hanlab.mit.edu/projects/sparserefine) | Selects high-entropy pixels from a dense low-resolution prediction, computes sparse high-resolution refinements and combines them through a gated ensembler. Selective local refinement, using a selector, and gating correction into a base prediction are established concepts. Our scope keeps resolution/encoder fixed and must distinguish action-effect information from ordinary selection with matched inputs/capacity and actual execution cost. |
| [PPN, AAAI2020 author abstract](https://ojs.aaai.org/index.php/AAAI/article/view/6926) | Uses a classification proposal network to distinguish critical patches for global/local refinement. Merely predicting where to refine or combining global and local predictions is insufficient as a new method claim. A matched ordinary local gate remains necessary. Only the abstract was inspected. |
| [From Failure Prediction to Component Selection, v1](https://arxiv.org/html/2610.16058v1), sections3.2–3.3 and4.3 | Distinguishes forecast quality, additional information and selection outcomes; matches retention budgets and learner families. Conditional-risk ranking is established; replacing unknown risks by fitted estimates does not inherit its guarantee. This reinforces fixed-input controls and end-to-end selection evaluation. The paper's inspected target concerns failure/retention of existing components, whereas our proposed target concerns actual counterfactual intervention outcomes. That distinction alone does not establish novelty or effectiveness. |

The new question worth a small falsifiable probe is whether actual single-action
error transitions can be reused for joint interventions, and whether their
pre-action prediction supports better decisions than same-information local
gating or generic value regression. No first-use or exclusivity claim is made.
The probe must include an ordinary-GN/full-per-action-decoding-and-pasting control
with its full cost. If it achieves the same accuracy with acceptable cost, shared
reference statistics alone do not justify the proposed method.

Two searches (three queries each) and eight distinct direct text-resource
attempts were made. The oversized CMU PDF failed (reported57,817,973bytes), and
OpenReview returned a verification page. Neither entry was retried or bypassed;
the existing arXiv v3 HTML provided the normalization paragraph. No external
implementation, package, weights, image or dataset was installed or executed.
Search snippets also exposed related patch/refinement work; those uninspected
algorithms are not represented as fully reviewed. Literature saturation remains
unproven; this pass narrows claims before spending real compute.

Scope accounting note: the four-search limit was interpreted as tool requests;
two requests contained six query strings. If counted per query, this exceeds
four. Record that ambiguity explicitly; do not present the pass as exact
preregistered search compliance. The eight direct-resource attempt cap was met,
and no effectiveness or novelty conclusion depends on a nominal search count.
