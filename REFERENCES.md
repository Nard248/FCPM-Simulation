# References

Scientific references for the methods and physics implemented in the FCPM library.

## Liquid Crystal Physics

1. **de Gennes, P.G. & Prost, J.** (1993).
   *The Physics of Liquid Crystals* (2nd ed.). Oxford University Press.
   — Foundation for Frank elastic energy, nematic order, and director field description.

2. **Frank, F.C.** (1958).
   On the theory of liquid crystals. *Discussions of the Faraday Society*, 25, 19-28.
   https://doi.org/10.1039/df9582500019
   — Original formulation of the elastic free energy: splay (K1), twist (K2), bend (K3).

## FCPM Technique

3. **Smalyukh, I.I., Shiyanovskii, S.V., & Lavrentovich, O.D.** (2001).
   Three-dimensional imaging of orientational order by fluorescence confocal polarizing microscopy.
   *Chemical Physics Letters*, 336(1-2), 88-96.
   https://doi.org/10.1016/S0009-2614(00)01471-8
   — Introduces FCPM for 3D director field reconstruction. Polarization-dependent intensity: I(alpha) proportional to [n . e(alpha)]^4.

4. **Smalyukh, I.I.** (2022).
   Liquid crystal colloids. *Annual Review of Condensed Matter Physics*, 13, 217-244.
   https://doi.org/10.1146/annurev-conmatphys-031620-110150
   — Review of FCPM applications in colloidal systems and topological structures.

## Liquid Crystal Solitons and Topological Structures

5. **Ackerman, P.J. & Smalyukh, I.I.** (2017).
   Diversity of knot solitons in liquid crystals manifested by linking of preimages in torons and hopfions.
   *Physical Review X*, 7(1), 011006.
   https://doi.org/10.1103/PhysRevX.7.011006
   — Torons, hopfions, and topological solitons in chiral nematic LCs.

6. **Smalyukh, I.I., Lansac, Y., Clark, N.A., & Trivedi, R.P.** (2010).
   Three-dimensional structure and multistable optical switching of triple-twisted particle-like excitations in anisotropic fluids.
   *Nature Materials*, 9(2), 139-145.
   https://doi.org/10.1038/nmat2592
   — Cholesteric fingers and triple-twisted structures.

7. **Tai, J.-S.B. & Smalyukh, I.I.** (2020).
   Three-dimensional crystals of adaptive knots. *Science*, 365(6460), 1449-1453.
   https://doi.org/10.1126/science.aaz3041
   — 3D solitonic crystals in chiral nematic LCs.

## Sign Optimization Algorithms

8. **Boykov, Y. & Kolmogorov, V.** (2004).
   An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision.
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(9), 1124-1137.
   https://doi.org/10.1109/TPAMI.2004.60
   — Graph cuts for binary labeling; used in `GraphCutsOptimizer`.

9. **Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P.** (1983).
   Optimization by simulated annealing. *Science*, 220(4598), 671-680.
   https://doi.org/10.1126/science.220.4598.671
   — Simulated annealing; used in `SimulatedAnnealingOptimizer`.

10. **Wolff, U.** (1989).
    Collective Monte Carlo updating for spin systems.
    *Physical Review Letters*, 62(4), 361-364.
    https://doi.org/10.1103/PhysRevLett.62.361
    — Wolff cluster algorithm; used for cluster moves in simulated annealing.

11. **Pearl, J.** (1988).
    *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann.
    — Belief propagation on factor graphs; used in `BeliefPropagationOptimizer`.

12. **Yedidia, J.S., Freeman, W.T., & Weiss, Y.** (2003).
    Understanding belief propagation and its generalizations.
    *Exploring Artificial Intelligence in the New Millennium*, 8, 236-239.
    — Loopy belief propagation analysis; relevant to convergence properties.

## Q-tensor Methods

13. **Schiele, K. & Trimper, S.** (1983).
    On the elastic constants of a nematic liquid crystal.
    *Physica Status Solidi (b)*, 118(1), 267-274.
    https://doi.org/10.1002/pssb.2221180132
    — Q-tensor representation Q_ij = S(n_i n_j - delta_ij / 3) for nematic order.

## Numerical Methods

14. **Lam, L. & Suen, S.Y.** (1997).
    Application of majority voting to pattern recognition: an analysis of its behavior and performance.
    *IEEE Transactions on Systems, Man, and Cybernetics*, 27(5), 553-568.
    — Majority voting principle; used in layer propagation for sign consistency.

15. **Briggs, W.L., Henson, V.E., & McCormick, S.F.** (2000).
    *A Multigrid Tutorial* (2nd ed.). SIAM.
    — Multigrid / hierarchical methods; inspiration for `HierarchicalOptimizer`.
