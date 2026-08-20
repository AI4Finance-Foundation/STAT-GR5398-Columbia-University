import os
import json
import argparse
from tqdm import tqdm
from statistics import mean, variance, median, pstdev
import gc
import torch
import numpy as np

from utils import load_dataset
from llm_test.model_Runner import model_runner
from rouge_score import rouge_scorer


def compute_stats(values):
	if not values:
		return {}
	return {
		"count": len(values),
		"mean": float(mean(values)),
		"variance": float(variance(values)) if len(values) > 1 else 0.0,
		"std": float(pstdev(values)),
		"median": float(median(values)),
		"min": float(min(values)),
		"max": float(max(values))
	}


def evaluate(args):
	os.makedirs(args.output_dir, exist_ok=True)
	per_sample_file = getattr(args, 'per_sample_file', 'per_sample_rouge.json')
	summary_file = getattr(args, 'summary_file', 'summary_rouge.json')

	# Load dataset (uses utils.load_dataset from the project)
	dataset_list = load_dataset(args.dataset, from_remote=args.from_remote)
	ds = dataset_list[0].get(args.split, None)
	if ds is None:
		raise ValueError(f"Split '{args.split}' not found in dataset")

	# Optional filtering by date range and symbol
	if args.start_date or args.end_date or args.symbol:
		start = args.start_date or ""
		end = args.end_date or "~"  # tilde sorts after dates
		symbol = args.symbol

		def _filter(x):
			ok = True
			if symbol:
				ok = ok and (x.get('symbol', None) == symbol)
			if start:
				ok = ok and (start <= x.get('period', ''))
			if end:
				ok = ok and (x.get('period', '') <= end)
			return ok

		ds = ds.filter(_filter)
		print(f"=== Filtered Dataset Length: {len(ds)} ===")

	total = len(ds)
	if args.max_samples and args.max_samples > 0:
		total = min(total, args.max_samples)

	# Instantiate model
	device = args.device if args.device else ("cuda:0" if args.use_cuda else "cpu")
	model = model_runner(args.model_name, cache_dir=args.cache_dir, device=device, fine_tuned_path=args.fine_tuned_path)

	scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

	per_sample = []
	rouge1_list = []
	rouge2_list = []
	rougeL_list = []
	batch_size = getattr(args, 'batch_size', 8)

	for i in tqdm(range(0, total, batch_size), desc="Generating and scoring"):
		batch_items = [ds[j] for j in range(i, min(i + batch_size, total))]
		prompts = [item.get('prompt') if isinstance(item, dict) else item['prompt'] for item in batch_items]
		gts = [item.get('answer') if isinstance(item, dict) else item['answer'] for item in batch_items]

		try:
			generated_list = model.generate_batch(prompts, max_length=args.max_length, temperature=args.temperature)
		except Exception as e:
			print(f"Batch generation error: {e}")
			generated_list = [f"<generation_error: {e}>"] * len(prompts)

		# Score
		try:
			score = scorer.score(gt, generated)
			r1 = score['rouge1'].fmeasure
			r2 = score['rouge2'].fmeasure
			rL = score['rougeL'].fmeasure
		except Exception:
			r1 = r2 = rL = 0.0

		for j, (item, prompt, gt, generated) in enumerate(zip(batch_items, prompts, gts, generated_list)):
			try:
				score = scorer.score(gt, generated)
				r1 = score['rouge1'].fmeasure
				r2 = score['rouge2'].fmeasure
				rL = score['rougeL'].fmeasure
			except Exception:
				r1 = r2 = rL = 0.0
			
			per_entry = {
				"index": i,
				"period": item.get('period', None),
				"symbol": item.get('symbol', None),
				"prompt": prompt,
				"ground_truth": gt,
				"generated": generated,
				"rouge1": r1,
				"rouge2": r2,
				"rougeL": rL
			}
			per_sample.append(per_entry)

			rouge1_list.append(r1)
			rouge2_list.append(r2)
			rougeL_list.append(rL)

	# Save per-sample results
	per_sample_path = os.path.join(args.output_dir, per_sample_file)
	with open(per_sample_path, 'w', encoding='utf-8') as f:
		json.dump(per_sample, f, ensure_ascii=False, indent=2)

	# Compute summary statistics
	summary = {
		"rouge1": compute_stats(rouge1_list),
		"rouge2": compute_stats(rouge2_list),
		"rougeL": compute_stats(rougeL_list)
	}

	summary_path = os.path.join(args.output_dir, summary_file)
	with open(summary_path, 'w', encoding='utf-8') as f:
		json.dump(summary, f, ensure_ascii=False, indent=2)

	print(f"Saved per-sample results to: {per_sample_path}")
	print(f"Saved summary stats to: {summary_path}")
	del model 
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	print(f"Memory usage after cleaning: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
	return per_sample_path, summary_path


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--dataset', default='dow30-202305-202405')
	p.add_argument('--from_remote', action='store_true')
	p.add_argument('--split', default='test')
	p.add_argument('--start_date', default=None)
	p.add_argument('--end_date', default=None)
	p.add_argument('--symbol', default=None)
	p.add_argument('--max_samples', type=int, default=0)
	p.add_argument('--model_name', default='Qwen/Qwen2.5-7B-Instruct')
	p.add_argument('--fine_tuned_path', default=None)
	p.add_argument('--cache_dir', default='./pretrained-models')
	p.add_argument('--device', default=None)
	p.add_argument('--use_cuda', action='store_true')
	p.add_argument('--max_length', type=int, default=2048)
	p.add_argument('--temperature', type=float, default=0.7)
	p.add_argument('--output_dir', default='./rouge_results')
	p.add_argument('--per_sample_file', default='per_sample_rouge.json')
	p.add_argument('--summary_file', default='summary_rouge.json')
	p.add_argument('--batch_size', default=8, type=int)
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	evaluate(args)
	

