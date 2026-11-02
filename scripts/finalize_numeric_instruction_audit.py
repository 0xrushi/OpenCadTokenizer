"""Publish manual review of the fixed numeric-caption audit; no model execution."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path('artifacts/numeric-instruction-audit-v1')
# Reviewed against per_case.json, not a classifier inferred from caption keywords.
# H: structural/relative-dimensional contradiction; M: canonical SEM-local reading.
# U: cannot certify the full instruction without units/frame/reference definitions.
REVIEWS = {
    '61281': ('U', 'Four circular cuts exist. Raw radii are 1, 1, 1.25, 1.25; centers are (5,16), (5,46), (57,16), (57,46). Quantization, units and depth prevent certification. Exclude the auxiliary Boolean material-difference probe: it violates cut-only monotonicity.'),
    '65220': ('U', 'A new annulus is wider but shorter than the source cylinder (scale 11 versus 7). Axial code midpoints differ by -13, but the axis is world +X; smaller/below and units lack a defined convention.'),
    '108938': ('H', 'The rectangular cut spans raw x=5..57 and y=14..48: 52 by 34, not 52 by 43. Uniform sketch scaling preserves this incompatible aspect ratio.'),
    '61197': ('U', 'A smaller-radius annulus is added (scale 12 versus 63). Planes change from 31..36 to 31..47 with a translation difference. The 9-unit below reference is unspecified.'),
    '62716': ('U', 'The added annulus has scale 6 versus source 2, but axial span 4 versus 32. It is wider and shorter; smaller and 12 units below are not uniquely defined.'),
    '106824': ('U', 'The new circle has raw radius 22 and sketch scale code 18. Code 18 is not itself a diameter; no physical-unit convention is provided to certify the claimed diameter.'),
    '61426': ('U', 'The cut has axial codes 12..31, span 19. This supports a depth reading in axial grid units, but below has no stated reference and the physical unit is unspecified.'),
    '102927': ('M', 'Under SEM-local coordinates the circle center is (31,31), radius 22; (31,53) and 9 are boundary coordinates. The claimed center/radius do not match that reading; arbitrary unstated coordinate/unit mappings are not ruled out.'),
    '92414': ('H', 'The second additive rectangular feature is replaced by a filled additive circle. There is no circular cut or inner loop creating the requested central hole.'),
    '71091': ('U', 'The new rectangular extrusion has axial span 14-11=3, supporting height in axial grid units. Top and the physical unit/frame remain undefined.'),
    '67468': ('H', 'An additive annulus lies wholly within the original filled cylinder with unchanged planes, translation and offsets. Kernel source/target volumes agree to 6e-14 and both material differences are zero: no new feature or 7-unit displacement.'),
    '72615': ('U', 'A cylinder fills the bottom of an existing annulus. Its upper plane is 26 grid bins below the original upper plane, but its midpoint is 13 bins below. Below lacks a reference point/unit.'),
    '62503': ('H', 'Target appends a filled additive cylinder to the cuboid, not a hole. The material probe finds positive added volume (0.0069428), zero removed volume.'),
    '72087': ('H', 'Target appends a filled additive cylinder without an inner loop, not a hole. Added volume is 0.026426 with zero removed volume; scale code 10 is not a radius.'),
    '67128': ('M', 'Under SEM-local/grid coordinates the cut circle has center (31,31), diameter 44 and axial span 31-25=6, not center (31,53), diameter 18 and height 7. Seven is the sketch scale code. Unit/frame ambiguity limits confidence.'),
    '71997': ('H', 'Target appends a filled additive cylinder to the polygonal prism, not a hole. Added material is positive (about 0.000024), removed material zero.'),
    '111243': ('H', 'The second additive rectangular feature becomes a filled additive cylinder, without a circular cut or inner loop. This does not create the requested cylindrical hole.'),
    '106128': ('H', 'The alleged diameter-12 hole is a straight-edged square cut with raw sides 44 and sketch scale code 12. There are no circle/arc primitives in the target.'),
    '73258': ('H', 'The new annulus has the same outer radius as the original cylinder, less volume and axial planes 31..47 versus original 0..31, along +Z with unchanged transform. It is not a larger base below the source; four is the raw radial gap 22-18.'),
    '73504': ('H', 'Target adds a filled D-shaped profile (line plus semicircular arc), not a cylindrical hole. Added volume is 0.072458 with zero removed volume; 14 is the sketch scale code.'),
    '66881': ('U', 'Target cuts an annular groove with outer/inner raw radii 22/18. Four equals radial thickness, but a four-unit displacement below the center is not defined by the caption.'),
    '103296': ('U', 'A lower filled cylinder becomes a thinner annulus (raw radii 22/15). Seven is radial thickness. Units/reference for seven units below the center are absent.'),
    '61181': ('U', 'The cut comprises two semicircular faces forming a full circular profile of raw radius 22, sketch scale 4. There is no supplied unit mapping certifying diameter 18.'),
    '103046': ('U', 'The circular cut has axial span 44-18=26, supporting height in axial grid units. Center, frame and physical-unit convention remain unspecified.'),
    '105500': ('H', 'The matched short cylinder axial span changes from 31-27=4 to 31-11=20: height INCREASES by 16 grid bins despite reduce its height by 16. Uniform coordinate conversion preserves the sign.'),
    '91704': ('H', 'Only the circular cut sketch scale changes, 4 to 3. Radius shrinks 25% with unchanged plane bounds, translation and offsets; this is an unrequested resize, not the stated move.'),
    '93257': ('H', 'A circular cut becomes rectangular; axial span decreases from 10 to 3 rather than increasing by nine. Sketch scale increases from 15 to 24, which is not extrusion height.'),
    '96987': ('H', 'Cut axial span correctly decreases by six (31 to 25), but sketch scale 9 to 12 also enlarges radius by one third, an unrequested resize. The edited feature is a cut, not an added solid cylinder.'),
    '109302': ('H', 'Circular extrusion axial span changes from one to 32 grid bins. Height increases by 31 instead of preserving dimensions under a rigid upward move.'),
    '93641': ('H', 'Cut sketch scale 57 to 9 shrinks radius to 9/57 of its source value; axial span also decreases 10 to 8. Translation and offsets are unchanged. This is not a rigid move two units right.'),
    '93672': ('H', 'The lower second prism is unchanged. The first sketch loses an inner loop and scale changes 41 to 37, with an offset change. These are topology/size edits rather than the stated move. Source kernel invalid: classification relies on program structure, not a certified source solid.'),
    '96023': ('H', 'The circular cut radius shrinks with scale 24 to 10, axial span changes 13 to 14 and offset changes. A rigid move cannot account for these unrequested dimension changes.'),
}


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    rows = json.loads((ROOT / 'per_case.json').read_text())
    parity = json.loads((ROOT / 'parser_parity.json').read_text())
    assert len(rows) == 32
    assert set(REVIEWS) == {r['example_id'].split(':')[-1] for r in rows}
    assert all(r['raw_to_prepared_exact'] for r in rows)
    assert all(r['raw_method'] == 'sequence' for r in rows)
    assert parity['checked_sequences'] == 64 and parity['max_absolute_difference'] < 1e-6
    reviews = []
    for row in rows:
        label, reason = REVIEWS[row['example_id'].split(':')[-1]]
        reviews.append({
            'example_id': row['example_id'], 'instruction': row['instruction'],
            'category': 'underspecified' if label == 'U' else 'inconsistent',
            'contradiction_confidence': {'H': 'high', 'M': 'moderate', 'U': None}[label],
            'reason': reason, 'source_kernel_status': row['source_geometry']['status'],
            'target_kernel_status': row['target_geometry']['status'],
            'auxiliary_material_probe_usable': row['solid_measurements'] is not None,
            'evidence_file': 'per_case.json',
        })
    # The Boolean probe for this cut-only pair returned entire solids on both sides.
    # Preserve raw measurements, but never use these impossible deltas as evidence.
    next(r for r in reviews if r['example_id'].endswith(':61281'))['auxiliary_material_probe_usable'] = False
    counts = Counter(r['category'] for r in reviews)
    high = [r['example_id'] for r in reviews if r['contradiction_confidence'] == 'high']
    moderate = [r['example_id'] for r in reviews if r['contradiction_confidence'] == 'moderate']
    assert counts == {'inconsistent': 20, 'underspecified': 12}
    assert len(high) == 18 and len(moderate) == 2
    write_json('reviewed_cases.json', reviews)
    with (ROOT / 'classification.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
        writer.writeheader(); writer.writerows(reviews)
    write_json('quarantine_candidates.json', {
        'action': 'Review candidates only; no dataset rows changed or excluded automatically.',
        'high_confidence_inconsistent': high, 'coordinate_unit_review_required': moderate,
        'underspecified': [r['example_id'] for r in reviews if r['category'] == 'underspecified'],
    })
    result = {
        'status': 'complete', 'cases': len(rows), 'categories': dict(counts),
        'inconsistent_confidence': {'high': len(high), 'moderate': len(moderate)},
        'raw_to_prepared_exact': 32, 'sequence_generated_captions': 32,
        'source_kernel_valid': sum(r['source_geometry']['status'] == 'valid' for r in rows),
        'target_kernel_valid': sum(r['target_geometry']['status'] == 'valid' for r in rows),
        'author_parity_sequences': parity['checked_sequences'],
        'author_parity_circles': parity['circles_checked'],
        'author_parity_max_absolute_difference': parity['max_absolute_difference'],
        'excluded_auxiliary_material_probe': ['cad_editor:train:61281'],
        'full_author_parser_rejections': parity['full_author_parser_rejections'],
        'interpretation': 'Targeted single-reviewer diagnostic sample, not a dataset prevalence estimate. Underspecified is not a certified incorrect label. Moderate flags assume canonical SEM-local/grid conventions.',
        'next_step': 'Review/curate existing numeric supervision and freeze an independently checked instruction-faithfulness panel before more model optimization. Keep current router.',
        'model_execution': False, 'training_launched': False, 'dataset_modified': False,
        'held_out_test_used': False,
    }
    write_json('results.json', result)
    table = '\n'.join('| ' + ' | '.join([
        r['example_id'].split(':')[-1], r['category'] + (f" ({r['contradiction_confidence']})" if r['contradiction_confidence'] else ''),
        r['instruction'], r['reason'],
    ]) + ' |' for r in reviews)
    report = '''# Numeric instruction audit — completed 2026-10-27

The existing numeric editing supervision contains concrete caption/target contradictions. Among these 32 selected training examples, manual review flags 18 high-confidence inconsistencies, two additional coordinate/unit-dependent inconsistencies and 12 underspecified instructions. This is a targeted diagnostic sample, not an estimated error rate for the dataset. None of the 12 underspecified cases is certified wrong merely because it lacks physical units.

## Scope and chain of evidence

Selection was fixed at 24 Add and eight Move captions containing numerals from the original pilot training subset, seed 542. All 32 have upstream method `sequence`. Instructions and both source/target sequences match the raw upstream records exactly in all 32 cases. No held-out test examples, training, inference, automatic relabeling or dataset exclusions were used.

The independent numeric trace matches the pinned author's dequantization/scale/translation/offset and 81 effective circle radii across all 64 sequences: maximum absolute difference 1.2186e-7 (tolerance 1e-6). This supports the local numeric conversion for these checks; it does not prove complete equivalence of Boolean/meshing backends. CadQuery/OCP reconstruction returns 31 valid source geometries and 32 valid targets. Source 93672 is invalid; its review uses explicit program topology changes rather than a certified source solid. Validity here does not imply one connected solid or instruction correctness.

The untouched author's full parser rejects five arc-containing sequences in the current NumPy 2.5.4 environment: its two-dimensional `np.cross` call raises a dimensionality ValueError. These environment compatibility failures are recorded in `parser_parity.json`; they are not evidence of bad captions or failed numeric parity. No upstream parser patch was applied to make these full parses succeed.

## Coordinates and dimensions

For six-bit codes q, sketch coordinates/axial planes/translation decode as d(q)=-1+2q/63; sketch scale is s=1.4*q_scale/63; offsets decode as o=-0.9+1.8*q_offset/63. With sketch axes x,y and normal z, a sketch point maps to T+(s*d(u)+o_x)*x+(s*d(v)+o_y)*y+plane*z. Circle raw radius is the mean half-span of the two diameters; its effective radius is raw_radius*(2/63)*s. Extrusion height is (upper_code-lower_code)*2/63, independent of sketch scale.

These are normalized geometry units, not guaranteed millimetres. A scale code of 18 is not itself a diameter of 18; a circle boundary point (31,53) is not its center. To avoid inventing missing conventions, unit-only disagreements remain underspecified unless an invariant such as aspect ratio, radius ratio, shape topology or increase/decrease direction establishes a contradiction. Two additional inconsistent cases are explicitly qualified by their SEM-local/grid interpretation.

## Concrete failures

- **105500:** “reduce its height by 16” changes axial span 4 to 20: an increase of 16 grid bins, with sign preserved by dequantization.
- **93641:** “Move central hole 2 units right” shrinks radius by a factor 57/9=6.33 and changes depth, with unchanged translation/offsets.
- **62503 / 72087 / 71997:** requested holes become filled additive cylinders. Material probes confirm positive added volume and zero removed material.
- **108938:** a requested 52-by-43 rectangular hole is 52-by-34 in sketch coordinates. Uniform scaling cannot fix the aspect ratio.
- **67468:** the new additive annulus is wholly inside the original solid; volumes agree to about 6e-14 and material differences are zero. The supposed new displaced feature does not change the shape.

Some numeric clauses do match a raw-grid convention: height three in 71091, depth 19 in 61426, height 26 in 103046. These partial matches do not certify the complete instructions, their physical units or their positional references.

## Likely source and limits

The pinned [upstream caption program](https://github.com/microsoft/CAD-Editor/blob/3363b99eb5d2c833c3fe963754f94c30cf341f92/data/caption_sequence.py) uses LLaMA-3-70B-Instruct to summarize sequence edits. Its prompt describes integer CAD fields and a sketch scaling parameter but does not state the dequantization ranges/unit convention implemented in the [upstream parser](https://github.com/microsoft/CAD-Editor/blob/3363b99eb5d2c833c3fe963754f94c30cf341f92/utils/parse_seq2obj.py). The observed captions are consistent with confusing scale codes, circle boundary coordinates and physical dimensions. That is a plausible failure mechanism, not proof of how the caption model reasoned.

This single-reviewer audit concentrates on numeric Add/Move examples and includes repeated geometric patterns. Counts cannot be extrapolated to all CAD-Editor data. Ambiguous captions may omit operations intentionally; even so, they cannot reliably supervise precise instruction execution. These findings do not absolve the model's earlier failures or establish a new quality gain. Exact target-program matching measures reproduction of the provided target, which can disagree with faithful execution of the caption.

One auxiliary Boolean probe is excluded: case 61281 preserves the source operation and appends only cuts, yet independent STEP differences returned essentially each entire solid on both sides. That violates cut-only monotonicity. Raw evidence is retained; those material deltas are not used in review. Its valid source/target reconstructions and numeric traces remain available. No quantitative conclusions rely on that failed probe.

## Per-case review

High confidence means a structural or dimension-invariant contradiction. Moderate confidence assumes canonical SEM-local/grid interpretation. Underspecified means a complete faithful edit cannot be certified from the caption and supplied conventions; it is not counted as a proven error.

| Training ID | Review | Original instruction | Evidence and qualification |
|---|---|---|---|
''' + table + '''

## Next action

Keep the existing generation-specialist/original-editing router. Pause optimization against unverified numeric captions. Review existing data first: use the 18 high-confidence IDs as initial review candidates; resolve the two coordinate-dependent flags separately; supply explicit units/frame/reference semantics for any retained numeric examples. Freeze a small independently checked instruction-faithfulness panel before running another learnability experiment. Do not silently replace a published caption with our own new caption, delete these rows, or reinterpret old benchmark scores.

This audit is finished. The next curation/review stage has not been launched. `quarantine_candidates.json` is a review list, not an applied training filter. A second human review or an independently validated verifier is needed before treating these diagnoses as authoritative corpus labels.

## Artifacts and reproduction

- `manifest.json`: fixed IDs, seed, original source/database hashes, image ID and evidence-stage code hash.
- `per_case.json` / `evidence.md`: original captions/SEMs, raw/dequantized primitives, geometry cache/export paths and measurements.
- `parser_parity.json`: 64 conversion comparisons, 81 circle radii, full-parser compatibility errors.
- `reviewed_cases.json`, `classification.csv`, `quarantine_candidates.json`, `results.json`: explicit manual judgments and machine-readable counts.
- `upstream/sources.json`: immutable upstream commit URLs and content hashes.
- `review_provenance.json`: evidence/report hashes and finalizer code hash.

Evidence was generated CPU-only in the cached Unsloth image via `CAD_CPU=1 ./scripts/cad.sh numeric-instruction-audit`; author parity via `CAD_CPU=1 ./scripts/cad.sh numeric-audit-parity`. Pure report assembly uses `python scripts/finalize_numeric_instruction_audit.py`. Preserve the evidence manifest: code changes after evidence collection can deliberately reject a main-stage resume with a provenance mismatch. Use a fresh output location for new evidence rather than replacing the original manifest.
'''
    (ROOT / 'findings.md').write_text(report)
    write_json('state.json', {'status': 'complete', 'cases': 32, 'classification': 'complete',
                            'report': str(ROOT / 'findings.md'), 'next_stage': 'data_review_not_started'})
    paths = [ROOT / n for n in ['manifest.json', 'per_case.json', 'parser_parity.json',
             'reviewed_cases.json', 'classification.csv', 'results.json', 'findings.md']]
    paths += [Path(__file__), Path('cadtokenizer/instruction_audit.py'), Path('cadtokenizer/instruction_parity.py')]
    write_json('review_provenance.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
