from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import unittest

from .common import read_config, require_container, write_json


def main():
    parser = argparse.ArgumentParser(description='CAD training/inference in the authorized local or Vast.ai runtime')
    parser.add_argument('command', choices=['doctor', 'prepare', 'train-vq', 'vq-profile', 'train-align', 'encode', 'train-sft', 'generate', 'evaluate', 'test', 'smoke',
                                          'benchmark-setup', 'benchmark-doctor', 'benchmark-test', 'benchmark-freeze',
                                          'benchmark-predict', 'benchmark-inference-profile', 'benchmark-score', 'benchmark-lenient', 'benchmark-judging', 'benchmark-smoke',
                                          'tokenizer-prepare', 'tokenizer-experiment', 'tokenizer-overfit', 'tokenizer-learning-curve',
                                          'tokenizer-profile', 'tokenizer-full', 'tokenizer-regression', 'tokenizer-continuation',
                                          'tokenizer-heldout', 'tokenizer-coverage-eval', 'sft-profile', 'tokenizer-coverage-hybrid', 'tokenizer-topk-check', 'tokenizer-loop-full', 'tokenizer-shape-audit', 'tokenizer-origin-ablation',
                                          'tokenizer-repeatability', 'lossless-prepare', 'lossless-sft', 'lossless-compare', 'lossless-editing-eval', 'lossless-train-recall', 'lossless-continue', 'lossless-grpo', 'lossless-experts', 'lossless-generation-eval', 'lossless-router-benchmark', 'lossless-router-memory-check', 'lossless-route', 'lossless-demo', 'lossless-numeric-ablation', 'numeric-instruction-audit', 'numeric-audit-parity', 'numeric-curation', 'addition-curation', 'cad-source-inspect', 'step-adapter-test', 'step-edit', 'step-adapter-expand', 'structured-bore-prepare', 'structured-bore-infer', 'structured-bore-evaluate', 'step-intent-edit', 'structured-bore-text-infer', 'structured-bore-text-evaluate', 'tokenizer-scale-origin-encode', 'tokenizer-scale-origin-decode', 'tokenizer-scale-origin-evaluate', 'hybrid-encode', 'hybrid-decode'])
    parser.add_argument('--config', default='configs/train.json')
    parser.add_argument('--tokenizer-checkpoint', default='artifacts/tokenizer-low-lr-v1/curve/checkpoint-40000.pt', help='Saved VQ checkpoint for hybrid-encode/decode')
    parser.add_argument('--resume', help='Checkpoint file for VQ/alignment, or Trainer checkpoint directory for SFT')
    parser.add_argument('--limit', type=int, help='Preparation records per source/split; evaluation pairs')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace a prepared/encoded database')
    parser.add_argument('--instruction')
    parser.add_argument('--source', help='Original CAD SEM text file, for editing')
    parser.add_argument('--output', default='artifacts/generated.json')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--cad-decoder', choices=['alignment', 'codebook'], help='Learned reverse projection or exact VQ codebook lookup')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--unconstrained', action='store_true')
    parser.add_argument('--directory', default='artifacts/smoke')
    parser.add_argument('--benchmark-config', default='configs/benchmark.json')
    parser.add_argument('--experiment-config', default='configs/tokenizer-local.json')
    parser.add_argument('--overfit-config', default='configs/tokenizer-overfit.json')
    parser.add_argument('--learning-curve-config', default='configs/tokenizer-learning-curve.json')
    parser.add_argument('--full-config', default='configs/tokenizer-full-ce.json')
    parser.add_argument('--regression-config', default='configs/tokenizer-regression.json')
    parser.add_argument('--continuation-config', default='configs/tokenizer-low-lr.json')
    parser.add_argument('--heldout-config', default='configs/tokenizer-heldout.json')
    parser.add_argument('--coverage-eval-config', default='configs/tokenizer-coverage-eval.json')
    parser.add_argument('--coverage-hybrid-config', default='configs/tokenizer-coverage-hybrid.json')
    parser.add_argument('--topk-config', default='configs/tokenizer-topk-check.json')
    parser.add_argument('--shape-audit-config', default='configs/tokenizer-shape-audit.json')
    parser.add_argument('--origin-ablation-config', default='configs/tokenizer-origin-ablation.json')
    parser.add_argument('--repeatability-config', default='configs/tokenizer-repeatability.json')
    parser.add_argument('--lossless-config', default='configs/lossless-sem.json')
    parser.add_argument('--lossless-sft-config', default='configs/lossless-sft-pilot.json')
    parser.add_argument('--pilot-mode', choices=['inspect','smoke','train'], default='train')
    parser.add_argument('--expert-arm', choices=['joint','generation','editing','report'], default='joint')
    parser.add_argument('--router', default='artifacts/lossless-routed-v2/router.json')
    parser.add_argument('--requests', help='Routed JSONL requests, or an input-only JSON request for step-edit')
    parser.add_argument('--inference-batch-size', type=int, help='Override the measured routed batch size')
    parser.add_argument('--numeric-arm', choices=['profile','parent','control','weighted','report'], default='control')
    parser.add_argument('--loop-full-config', default='configs/tokenizer-loop-full.json')
    parser.add_argument('--manifest', help='Frozen benchmark manifest JSON')
    parser.add_argument('--predictions', help='Benchmark predictions JSONL, with its matching .meta.json')
    parser.add_argument('--resume-predictions', action='store_true')
    parser.add_argument('--vlm-scores', help='Completed VLM score file exported by benchmark-judging')
    parser.add_argument('--human-scores', help='Completed blinded human ranking file')
    parser.add_argument('--baseline-report', action='append', help='NAME=path/to/metrics.json; supply two baselines for blinded human comparisons')
    args = parser.parse_args()
    require_container()
    if args.command in ('hybrid-encode', 'hybrid-decode'):
        from .hybrid_tokenizer import run_file
        run_file('encode' if args.command == 'hybrid-encode' else 'decode', args.source, args.output, args.tokenizer_checkpoint)
        return
    if args.command == 'tokenizer-scale-origin-encode':
        from .scale_origin_experiment import encode_study
        encode_study()
        return
    if args.command == 'tokenizer-scale-origin-decode':
        from .scale_origin_experiment import decode_study
        decode_study()
        return
    if args.command == 'tokenizer-scale-origin-evaluate':
        from .scale_origin_evaluation import evaluate_study
        evaluate_study()
        return
    if args.command == 'structured-bore-text-infer':
        from .structured_bore_inference import run_inference
        run_inference('artifacts/structured-bore-diagnostic-v2', text_only=True)
        return
    if args.command == 'structured-bore-text-evaluate':
        from .structured_bore_study import evaluate
        evaluate('artifacts/structured-bore-diagnostic-v2')
        return
    if args.command == 'step-intent-edit':
        from pathlib import Path
        import cadquery as cq
        from .structured_bore import parse as parse_intent, execute as execute_intent
        if not args.requests or not args.source or Path(args.output).suffix.lower() not in ('.step', '.stp'):
            parser.error('step-intent-edit requires --source source.step --requests intent.json --output output.step')
        if Path(args.source).resolve() == Path(args.output).resolve():
            parser.error('Output must differ from the source STEP')
        intent = parse_intent(Path(args.requests).read_text())
        shape = cq.importers.importStep(args.source).val()
        result = execute_intent(shape, intent)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cq.exporters.export(result, args.output)
        return
    if args.command == 'structured-bore-prepare':
        from .structured_bore_study import prepare
        prepare()
        return
    if args.command == 'structured-bore-infer':
        from .structured_bore_inference import run_inference
        run_inference()
        return
    if args.command == 'structured-bore-evaluate':
        from .structured_bore_study import evaluate
        evaluate()
        return
    if args.command == 'step-adapter-expand':
        from .step_bore_expansion import run_expansion
        run_expansion()
        return
    if args.command == 'step-edit':
        from pathlib import Path
        from .step_bore_adapter import worker
        if not args.requests or Path(args.output).suffix.lower() not in ('.step', '.stp'):
            parser.error('step-edit requires --requests input-only.json and --output output.step')
        request = read_config(args.requests)
        if Path(request['source_step']).resolve() == Path(args.output).resolve():
            parser.error('step-edit output must differ from its source STEP')
        worker(args.requests, args.output)
        return
    if args.command == 'step-adapter-test':
        from .step_adapter_study import run_study
        run_study()
        return
    if args.command == 'cad-source-inspect':
        from .cad_source_inspect import run_inspection
        run_inspection()
        return
    if args.command == 'addition-curation':
        from .addition_curation import run_curation
        run_curation()
        return
    if args.command == 'numeric-curation':
        from .numeric_curation import run_curation
        run_curation()
        return
    if args.command == 'numeric-audit-parity':
        from .instruction_parity import run_parity
        run_parity()
        return
    if args.command == 'numeric-instruction-audit':
        from .instruction_audit import run_audit
        run_audit()
        return
    if args.command == 'lossless-numeric-ablation':
        from .numeric_ablation import run_numeric
        run_numeric(args.numeric_arm,args.pilot_mode)
        return
    if args.command == 'lossless-demo':
        from .routed_demo import run_demo
        run_demo()
        return
    if args.command == 'lossless-router-memory-check':
        from .routed_memory_check import run_memory_check
        run_memory_check()
        return
    if args.command == 'lossless-router-benchmark':
        from .routed_release import run_routed_benchmark
        run_routed_benchmark(args.pilot_mode)
        return
    if args.command == 'lossless-route':
        from .routed_release import routed_inference
        routed_inference(args.router,args.instruction,args.source,args.requests,args.output,args.inference_batch_size)
        return
    if args.command == 'lossless-generation-eval':
        from .generation_validation import run_generation_validation
        run_generation_validation()
        return
    if args.command == 'lossless-experts':
        from .task_experts import run_experts
        run_experts(args.expert_arm,args.pilot_mode)
        return
    if args.command == 'lossless-grpo':
        from .grpo import run_grpo
        run_grpo(args.pilot_mode)
        return
    if args.command == 'lossless-continue':
        from .lossless_continue import run_continuation
        run_continuation(args.pilot_mode)
        return
    if args.command == 'lossless-train-recall':
        from .lossless_train_recall import run_train_recall
        run_train_recall()
        return
    if args.command == 'lossless-editing-eval':
        from .lossless_editing_eval import run_editing_evaluation
        run_editing_evaluation()
        return
    if args.command == 'lossless-compare':
        from .lossless_compare import run_comparison
        run_comparison()
        return
    if args.command == 'doctor':
        import torch
        versions = {}
        for name in ['torch', 'numpy', 'transformers', 'peft', 'accelerate', 'unsloth', 'tokenizers']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = 'MISSING'
        report = {'runtime': os.environ.get('CADTOKENIZER_RUNTIME', 'unsloth'),
                  'image_id': os.environ.get('CADTOKENIZER_IMAGE_ID'), 'versions': versions,
                  'cuda_available': torch.cuda.is_available(),
                  'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        print(json.dumps(report, indent=2))
        write_json('.runtime/environment.json', report)
        if any(versions[x] == 'MISSING' for x in ('torch', 'numpy', 'transformers', 'peft', 'accelerate', 'tokenizers')):
            raise RuntimeError('Required packages are missing from the selected runtime')
        return
    if args.command in ('test', 'benchmark-test'):
        suite = unittest.defaultTestLoader.discover('tests', pattern='test_bench*.py' if args.command == 'benchmark-test' else 'test*.py')
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.command == 'benchmark-setup':
        from .bench_runtime import setup_evaluation
        setup_evaluation()
        return
    if args.command == 'benchmark-doctor':
        from .bench_runtime import evaluation_doctor
        evaluation_doctor()
        return
    config = read_config(args.config)
    if args.cad_decoder is not None:
        config.setdefault('inference', {})['cad_decoder'] = args.cad_decoder
    if args.command.startswith('benchmark-'):
        settings = read_config(args.benchmark_config)
        if args.command == 'benchmark-freeze':
            from .bench_data import freeze_manifest
            freeze_manifest(config, settings, args.manifest)
        elif args.command == 'benchmark-inference-profile':
            from .bench_inference_profile import run_inference_profile
            run_inference_profile(config, settings, args.manifest, [8, 16, 32, 64], args.limit or 64)
        elif args.command == 'benchmark-predict':
            from .bench_predict import benchmark_predict
            benchmark_predict(config, settings, args.manifest, args.predictions, args.resume_predictions, args.limit, batch_size=args.inference_batch_size)
        elif args.command == 'benchmark-lenient':
            from .bench_lenient import lenient_rescore
            lenient_rescore(settings, args.manifest)
        elif args.command == 'benchmark-score':
            from .bench_score import benchmark_score
            benchmark_score(config, settings, args.manifest, args.predictions, args.vlm_scores, args.human_scores)
        elif args.command == 'benchmark-judging':
            from .bench_judging import export_judging
            directory = None if args.directory == 'artifacts/smoke' else args.directory
            export_judging(settings, args.manifest, args.baseline_report, directory)
        elif args.command == 'benchmark-smoke':
            from .bench_smoke import benchmark_smoke
            directory = 'artifacts/benchmark-smoke' if args.directory == 'artifacts/smoke' else args.directory
            benchmark_smoke(config, settings, directory)
        return
    if args.command == 'tokenizer-loop-full':
        from .tokenizer_loop_full import run_loop_full
        run_loop_full(read_config(args.loop_full_config))
    elif args.command == 'lossless-sft':
        from .lossless_sft import run_lossless_sft
        run_lossless_sft(args.lossless_sft_config,args.pilot_mode)
    elif args.command == 'lossless-prepare':
        from .lossless_prepare import prepare_lossless
        prepare_lossless(read_config(args.lossless_config))
    elif args.command == 'tokenizer-repeatability':
        from .tokenizer_repeatability import run_repeatability
        run_repeatability(read_config(args.repeatability_config))
    elif args.command == 'tokenizer-origin-ablation':
        from .tokenizer_origin_ablation import run_origin_ablation
        run_origin_ablation(read_config(args.origin_ablation_config))
    elif args.command == 'tokenizer-shape-audit':
        from .tokenizer_shape_audit import run_shape_audit
        run_shape_audit(read_config(args.shape_audit_config))
    elif args.command == 'sft-profile':
        from .sft_profile import run_sft_profile
        run_sft_profile(config, config['sft_profile'])
    elif args.command == 'tokenizer-coverage-hybrid':
        from .tokenizer_coverage_hybrid import run_coverage_hybrid
        run_coverage_hybrid(read_config(args.coverage_hybrid_config))
    elif args.command == 'tokenizer-topk-check':
        from .tokenizer_topk_check import run_topk_check
        run_topk_check(read_config(args.topk_config))
    elif args.command == 'tokenizer-coverage-eval':
        from .tokenizer_coverage_eval import run_coverage_eval
        run_coverage_eval(read_config(args.coverage_eval_config))
    elif args.command == 'tokenizer-heldout':
        from .tokenizer_heldout import run_heldout
        run_heldout(read_config(args.heldout_config))
    elif args.command == 'tokenizer-continuation':
        from .tokenizer_continuation import run_continuation
        run_continuation(read_config(args.continuation_config))
    elif args.command == 'tokenizer-regression':
        from .tokenizer_regression import run_regression
        run_regression(config, read_config(args.regression_config))
    elif args.command in ('tokenizer-profile', 'tokenizer-full'):
        from .tokenizer_full import run_full
        run_full(config, read_config(args.full_config), profile_only=args.command == 'tokenizer-profile')
    elif args.command == 'tokenizer-learning-curve':
        from .tokenizer_learning_curve import run_learning_curve
        run_learning_curve(config, read_config(args.learning_curve_config))
    elif args.command == 'tokenizer-overfit':
        from .tokenizer_overfit import run_overfit
        run_overfit(config, read_config(args.overfit_config))
    elif args.command in ('tokenizer-prepare', 'tokenizer-experiment'):
        from .tokenizer_experiment import prepare_experiment, run_experiment
        settings = read_config(args.experiment_config)
        (prepare_experiment if args.command == 'tokenizer-prepare' else run_experiment)(config, settings)
    elif args.command == 'prepare':
        from .prepare import prepare
        prepare(config, args.limit, args.overwrite)
    elif args.command == 'train-vq':
        from .vq import train_vq
        train_vq(config, args.resume)
    elif args.command == 'vq-profile':
        from .vq_profile import run_profile
        run_profile(config)
    elif args.command == 'train-align':
        from .alignment import train_alignment
        train_alignment(config, args.resume)
    elif args.command == 'encode':
        from .encode import encode_geometry
        encode_geometry(config, args.overwrite)
    elif args.command == 'train-sft':
        from .sft import train_sft
        train_sft(config, args.resume)
    elif args.command == 'generate':
        if not args.instruction:
            parser.error('generate requires --instruction')
        from .inference import generate
        generate(config, args.instruction, args.source, args.output, args.max_new_tokens)
    elif args.command == 'evaluate':
        from .inference import evaluate_vq
        evaluate_vq(config, args.limit or 100, args.aligned, not args.unconstrained)
    elif args.command == 'smoke':
        from .smoke import smoke
        smoke(config, args.directory)


if __name__ == '__main__':
    main()
