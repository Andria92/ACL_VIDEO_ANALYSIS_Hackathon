# Responsible AI and Measurement Boundaries — V1

## Purpose

ACL Movement Analytics Lab is an exploratory research and educational tool for interpretable
markerless 2D analysis of observable movement in supplied football footage. It helps a human
reviewer inspect projected or body-normalized movement descriptors, their source frames, and the
evidence supporting or withholding each result.

It is not an autonomous clinical decision system.

## Intended use

- Review a human-selected movement window in a supplied video.
- Track a human-identified target athlete using markerless pose estimates.
- Quantify supported image-plane or normalized movement descriptors.
- Inspect measurement support, missing intervals, and exact source-frame provenance.
- Compare documented cases descriptively when independent-case counts and mutually supported
  features are sufficient.

Results require human or research interpretation in the context of the source footage and case
metadata.

## Out-of-scope uses

The software does not estimate or claim:

- ACL ligament load or ACL strain;
- ground-reaction force, joint torque, or joint moment;
- true 3D joint angles, true depth, or 3D biomechanics;
- a causal injury mechanism;
- injury diagnosis, prospective injury risk, or medical prognosis;
- clinical treatment or rehabilitation recommendations;
- population-normal, abnormal, safe, or dangerous movement classifications;
- athlete identity through face recognition;
- injured side or medical details from movement appearance.

It must not be used to make player-selection, return-to-play, treatment, insurance, or other
high-impact decisions.

## Measurement boundaries

Outputs describe observable image-plane geometry recovered from markerless pose estimates.
Degree-valued metrics are projected 2D angles or orientations, not anatomical 3D joint angles.
Body-normalized distances and rates remain image-plane descriptors; they are not metric pitch
distance, physical speed, force, or loading.

Camera projection can change a descriptor even when the underlying 3D movement is unchanged.
Multiple views therefore remain view-specific. The current synthesis chooses a preferred evidence
view per feature and does not average projected angles across cameras.

## Uncertainty and QC behavior

The existing QC pipeline uses system evidence such as target continuity, pose availability,
landmark confidence, rejected/interpolated landmarks, geometry status, robust-dynamic status,
path status, continuous frame coverage, and human-marked unavailable intervals.

User-facing states use the distinctions the current pipeline supports:

- **Supported**: the current target, landmark, geometry, and relevant dynamic/path rules support
  the displayed result.
- **Limited**: a sequence- or interval-level result has only partial supported coverage and must be
  interpreted with its evidence details.
- **Unavailable**: the required evidence is absent or rejected. No numeric measurement is shown.

Unsupported samples remain null. They are not converted to zero, imputed, bridged in trajectory
plots, or included in descriptive statistics. A single supported sample can support its observed
value, but it cannot support a change, frame-to-frame change, or sample standard deviation; those
outputs remain unavailable.

The software does not introduce a calibrated statistical uncertainty or demographic fairness
score. Labels describe measurement support under engineering QC rules, not diagnostic confidence.

## Similarity interpretation

Similarity is currently evidence-gated and unavailable until a sufficient human-validated case
library and mutually supported features exist.

If enabled, similarity means similarity within the measured movement representation used by the
engine. It does not mean:

- identical ACL injury mechanism;
- the same biological cause;
- equivalent tissue loading;
- the same diagnosis or clinical condition;
- causal equivalence.

Any future result must state the features used, the features missing or excluded, important
differences, and deterministic contributors from the actual similarity calculation. Missing or
low-support dimensions must not be treated as zero or silently dominate the comparison. UMAP
display distance must not be used as similarity. No LLM-generated feature attribution is used.

## Observation and interpretation

Research measurements are the observed/measured layer. The Movement Story is a deterministic
interpretive summary built only from supported or explicitly limited evidence under controlled
rules. It may describe projected movement change; it must not convert temporal association into
injury causation.

Acceptable: “The projected hip-knee-ankle angle decreased across the supported interval.”

Not acceptable: “The ACL ruptured because the knee collapsed.”

## Human oversight

Human review remains required for:

- video and case-source verification;
- target-athlete identity;
- injured-side metadata;
- target ROI corrections and unavailable intervals;
- selection of the movement window and any analysis-boundary change;
- camera cuts, occlusion, replay timing, and footage quality;
- interpretation and contextual research conclusions.

Unknown identity, injury confirmation, or laterality must remain unknown or not recorded. The
movement pipeline must not fill these fields from video appearance.

## Dataset and footage limitations

Broadcast and online footage is not collected as a calibrated biomechanical dataset. Performance
can vary with camera angle, resolution, athlete scale, motion blur, occlusion, lighting,
compression, kit/body/background contrast, partial visibility, camera motion, cuts, zoom,
broadcast framing, and replay speed. A replay of the same event is another view, not another
independent case.

Public availability of footage does not automatically grant unrestricted reuse, redistribution,
or ownership rights. Source references and known rights or reuse notes should be recorded for each
case; the application does not claim rights that have not been verified.

## Pose and model limitations

Markerless pose models can mislocalize landmarks, swap people, lose the target during overlap, or
fail on body poses and image conditions outside their training domain. COCO-17 pose output lacks
heel and foot-index landmarks, which limits foot and ankle interpretation. Smoothing and short-gap
interpolation do not turn hidden joints into observations; their provenance remains visible.

## Bias and generalization limitations

Performance may vary across body morphology, skin/kit/background contrast, clothing, lighting,
competition broadcast style, camera placement, movement type, and other domains. This project has
not measured fairness or equivalent error rates across demographic groups and makes no such claim.
Broader, independently annotated validation is required before generalizing performance.

## Privacy and provenance

Each case should retain, where available:

- a video/source reference and rights or reuse note;
- an injury-confirmation source;
- the source of athlete identity and injured laterality;
- the human annotation session and annotator identifier;
- derived-analysis artifact identifiers and processing metadata.

The human research metadata file can retain `identity_source`, `injury_status`,
`injury_confirmation_source`, `video_source_reference`, and `video_rights_note`. Absent values
remain “not recorded.” Browser payloads expose safe file labels or data-root-relative artifact
references, not full local filesystem paths.

## Known risks

- A plausible 2D trajectory may still reflect camera projection or a wrong target.
- Partial coverage can make a limited summary appear more complete than it is unless the support
  panel is reviewed.
- Supplied case metadata may be wrong, incomplete, or weakly sourced.
- Small case libraries can encourage over-interpretation of descriptive patterns.
- A similarity score may be mistaken for medical or causal equivalence without the required
  qualifier and feature accounting.

## Future validation work

- Independent annotation agreement for target identity, ROI, unavailable intervals, movement
  windows, and laterality provenance.
- Error characterization by camera view, resolution, blur, occlusion, athlete scale, morphology,
  kit/background contrast, and model/domain conditions.
- Comparison with calibrated multi-camera or laboratory reference data only for explicitly matched
  projected quantities; do not silently expand to force, loading, or true 3D claims.
- Predefined missing-feature and minimum-overlap rules before enabling similarity.
- Case-library audits for source, injury-confirmation, rights, and annotation provenance.
- Review by biomechanics, sports-medicine, Responsible AI, privacy, and footage-rights experts.

## V1 human-review items

Scientific and project owners should review the controlled movement vocabulary, completeness
thresholds, registered case/injury sources, injured-side sources, footage rights notes, and the
criteria required before cross-case similarity is enabled.
