# pyfe3d-dev

SymPy scripts that generate the expanded element code of
[pyfe3d](https://github.com/saullocastro/pyfe3d). Each script under
`deriving_equations/<element>/` prints the sparsity pattern and the value
lines of one element matrix, which are then pasted into the corresponding
`.pyx`.

## Elements covered

`beamc`, `beamLR`, `beamcurvedLR`, `quad4r`, `tria3r`, `spring`, `truss`.

`quad4` is **not** covered any more. Its constitutive and internal-force
methods in `pyfe3d/quad4.pyx` were rewritten as explicit quadrature loops
over the `Quad4Probe` rows (`_update_probe_KC0ve`, `_update_probe_BL_G`,
`_update_probe_KCNLve`, `_update_probe_finte_nonlinear`), so there is no
generated block left for a generator to be the source of truth of. The
scripts had drifted out of step with the `.pyx` and were removed in 0.10.0
rather than kept as a misleading reference. `update_KG`,
`update_KG_given_stress`, `update_M` and the Piston theory methods of
`Quad4` are still expanded code, produced by those scripts before their
removal; recover them from the git history if they ever need regenerating.
