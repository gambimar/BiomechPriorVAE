# BiomechPriorVAE code review

The review covers the tracked Python, MATLAB and Bash source, the exploratory notebook, five inactive MATLAB autosaves, and the supplied model artifacts at baseline `e0e9d3e`. The main corrections concern training statistics, input gradients, contact mapping and batch error handling. The published weights and scientific model assets remain unchanged.

## Correctness fixes

| Area | Previous behavior | Correction |
| --- | --- | --- |
| Training scaler | Averaged batch means and within-batch variances, losing between-batch variance and weighting a short final batch equally | Fit population statistics over every selected training frame with `StandardScaler.partial_fit` |
| VAE loss | For `(batch, time, latent)` tensors, KL reduction summed the time axis and averaged latent coordinates | Sum the final latent axis and average frames; for a one-frame, 24-latent model this makes KL 24 times the old value at the same beta |
| Training APIs | Ignored dataset arguments, depended on a module-global feature list, used inconsistent defaults and called `.cpu()` on a decoder tuple | Use explicit paths/features, matching 46-feature defaults and decoder means |
| Dataset windows | Dropped the final two valid one-frame samples; multi-frame force assembly had incompatible shapes | Include every complete window and construct correctly shaped feature tensors |
| Dataset metadata | Selected ankle moments and right/left forces by fixed array position; some corrupt reads silently yielded omissions or zeros | Resolve joints and feet by name, sum contact bodies by side, and raise contextual errors on invalid reads |
| Inference loading | The bundled scaler unpickled CUDA storage before moving tensors to the requested device | Remap legacy tensor storage to CPU first; validate statistics and then transfer explicitly |
| Input gradients | Backpropagated into all model parameters; Gaussian variance flooring could produce a derivative inconsistent with the scored objective | Freeze parameters, request only input gradients, and differentiate an explicit variance clamp |
| Feature layouts | The current 46-feature training default was unsupported by the wrapper; several widths have ambiguous meanings | Support the current layouts and allow explicit ordered `state_keys`, preserving historical defaults |
| Global bridge | A failed initialization could leave a previous model active | Clear the previous model before attempting replacement |
| Simulation runner | MATLAB's failure exit code `1` also meant completion, so a crash could be reported as ten converged runs | Reserve process exit code `10` for completion and abort on MATLAB failures |
| Reference MATLAB bridge | Fixed state indices could split contacts as nine on the right and seven on the left, and gradients always assumed eight contacts per foot | Resolve named joint/contact metadata, support six or eight contacts per foot and split gradients at the actual input width |
| MATLAB setup | Missing prior parameters crashed the optional-prior branch; free-speed bounds had different orientations; standing and gait could use different contact models | Guard the prior, match bound shapes and share the contact-model factory |
| Export utilities | Executed dataset work on import, searched upward indefinitely outside a checkout, built autograd graphs and used fixed paths | Use import-safe command-line entry points, paths from `__file__`, inference mode and deterministic sampling |
| Conversion and display | Default B3D frame count raised an exception, incomplete joint mappings shortened outputs, and the pose viewer spun without sleeping | Read the remaining trial, reject incomplete mappings and yield the CPU |

Changes to the scaler, KL weighting, final-frame inclusion and contact features affect newly trained models. Retrain and save a new checkpoint/scaler pair together; attaching newly computed statistics to the existing checkpoint changes its meaning. Historical simulation results also need reassessment after correcting contact assignment.

## Performance improvements

Inference avoids constructing an autograd graph for objective-only calls and no longer allocates or accumulates parameter gradients during input differentiation. Training accumulates metrics on the training device and transfers them once per epoch, rather than synchronizing three scalar losses per batch. Model compilation is optional, worker count is configurable, and persistent workers can be enabled. The full-pose thumbnail export batches the selected feature rows into one encoder call. The reference MATLAB bridge skips joint-moment calculations for priors that do not consume moments, including the published 50-feature model, and only requests moment derivatives for gradient evaluations.

A local CPU measurement used the published 50-feature checkpoint, a fixed synthetic batch of 50 raw states, one Torch thread and 100 timed calls after warmup. With Python 3.13 and Torch 2.13 on Windows, median objective latency was **3.85 to 3.59 ms**, and input-gradient latency was **11.13 to 8.28 ms** against the baseline. These measurements cover Python inference, not MATLAB, IPOPT or an end-to-end gait simulation; laptop load affected timing in concurrent runs. Reproduce with `python benchmarks/benchmark_inference.py --repeats 100`.

## Verification and remaining limitations

The regression suite exercises synthetic training, strided dataset reads, named force/ankle mappings, worker serialization, published-checkpoint CPU inference, finite-difference input gradients, utility exports and mocked MATLAB process statuses. All **76 tests passed** on Windows with both Python 3.12.10/Torch 2.11 and Python 3.13.12/Torch 2.13; the Python 3.12 run disabled CUDA to verify CPU-only operation. Python syntax/undefined-name checks and Bash syntax checks also passed. Seventeen MATLAB fixtures are included under `tests/matlab`, but MATLAB, BioMAC, mexIPOPT and real `.b3d` training data were unavailable for full solver or dataset validation. Native Nimble rendering and VTK geometry conversion also need their respective runtimes.

The root MATLAB objective is a reference copy. Apply it to BioMAC's `@Collocation/vaeReconstructionTerm.m` and keep the new `vaeFeatureLayout`, `vaeStateIndices`, `vaeJointMoments` and `vaeStateGradient` helpers on the MATLAB path. The existing toolbox class copy is otherwise unchanged by this repository update. Contact/state and joint-moment APIs were checked against the toolbox's [Gait3d implementation](https://github.com/mad-lab-fau/BioMAC-Sim-Toolbox/blob/opensimContactModel/src/model/gait3d/@Gait3d/Gait3d.m). Moment-based priors use model mass and height for normalization; `vaeParams.momentScale = 1/(9.81*65.9*1.83)` retains the historical fixed factor when reproducing that convention.

All 22 supplied `.osim` files parse as XML and checked numeric parameters are finite. All 42 `.mat` files load, and their stored OpenSim hashes match the corresponding LF-normalized source. Windows CRLF checkouts change the raw bytes and may cause unnecessary cache regeneration in tools that hash those bytes. The checkpoint contains 20 finite tensors and loads strictly as input 50, latent 24, hidden 512; the scaler has matching finite vectors with positive scales.

Several follow-ups remain:

- **Validation leakage:** frame-level random splitting shares correlated subject/trial data across training and validation. Use subject-disjoint splits to evaluate generalization, and compare activity-specific held-out distributions.
- **Legacy notebook:** `notebook/latent_space_analysis.ipynb` targets missing 31-feature artifacts and treats the `(mean, variance)` decoder return as a tensor. Other cells assume equal batch sizes, use removed NumPy APIs or search indefinitely for `.git`. Use the maintained export scripts for the current model; updating the notebook needs a deliberate choice of model and analysis outputs.
- **Legacy arrays:** the tracked latent representations have 20 coordinates and interpolated poses have 33 features. They are finite but do not represent the bundled 24-latent, 50-feature model; regenerate and label them before using them for comparisons.
- **Historical MATLAB scripts:** autosaves are inactive and include incomplete code. Older experiment drivers retain machine-specific paths. Winter parallel helpers share intermediate standing results and need isolated output names and explicit per-run seeds before concurrent use. The COD driver requests missing unversioned weights/scaler, omits a required Python path and calls an external toolbox namespace; repairing it needs its external measurement data and intended checkpoint.
- **Unsupported MATLAB options:** `running3D` now rejects fixed-duration and tracking modes explicitly because its predictive-gait interface has no corresponding inputs. Reintroducing these modes needs an API and toolbox validation.
- **Dependencies:** separate optional dataset, visualization and notebook dependencies from inference, declare direct optional VTK/seaborn requirements and remove unused torchvision from the legacy requirements file. A dependency migration should update `uv.lock` together with those choices.
