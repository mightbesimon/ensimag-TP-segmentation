# status.json (global experiment_output/status.json, and the per-run
# <timestamp>/status.json) plus the project-root dashboard.html built from
# it -- a clickable overview of every run and every model hash ever trained.
# self-contained (no CDN, no fetch()) so the dashboard opens fine off disk.

import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP_FORMAT = '%y%m%dT%H%M%S'


class Status:
	@staticmethod
	def _global_path(out_root):
		return os.path.join(out_root, 'status.json')

	@staticmethod
	def read_global(out_root):
		path = Status._global_path(out_root)
		if not os.path.exists(path):
			return {'models': {}, 'runs': {}}
		with open(path) as f:
			return json.load(f)

	@staticmethod
	def write_global(out_root, status):
		with open(Status._global_path(out_root), 'w') as f:
			json.dump(status, f, indent='\t')

	@staticmethod
	def note_model(out_root, timestamp, run_spec, model_hash, checkpoints_out):
		status = Status.read_global(out_root)
		entry = status['models'].setdefault(model_hash, {
			'model_key': run_spec.model_key, 'class_name': run_spec.class_name,
			'enc_channels': list(run_spec.enc_channels), 'dec_channels': list(run_spec.dec_channels),
			'max_epoch_trained': 0, 'best_bce': None, 'best_epoch': None, 'runs': [],
		})
		for c in checkpoints_out:
			entry['max_epoch_trained'] = max(entry['max_epoch_trained'], c['epoch'])
			mean_bce = sum(r['bce'] for r in c['results']) / len(c['results'])
			if entry['best_bce'] is None or mean_bce < entry['best_bce']:
				entry['best_bce'], entry['best_epoch'] = mean_bce, c['epoch']
		entry['last_updated'] = datetime.now().strftime(TIMESTAMP_FORMAT)
		if timestamp not in entry['runs']:
			entry['runs'].append(timestamp)

		run_entry = status['runs'].setdefault(timestamp, {'created': timestamp, 'models': [], 'stage': 'training'})
		if model_hash not in run_entry['models']:
			run_entry['models'].append(model_hash)

		Status.write_global(out_root, status)

	@staticmethod
	def set_run_stage(out_root, timestamp, stage):
		status = Status.read_global(out_root)
		if timestamp in status['runs']:
			status['runs'][timestamp]['stage'] = stage
			Status.write_global(out_root, status)

	@staticmethod
	def write_run_status(run_dir, stage, model_runs):
		status = {
			'stage': stage,
			'models': {
				r.model_hash: {'label': r.run_spec.short_label, 'current_epoch': r.current_epoch}
				for r in model_runs
			},
		}
		with open(os.path.join(run_dir, 'status.json'), 'w') as f:
			json.dump(status, f, indent='\t')


class Dashboard:
	PATH = os.path.join(ROOT, 'dashboard.html')

	@classmethod
	def build(cls, output_root):
		status = Status.read_global(output_root)
		out_rel = os.path.relpath(output_root, ROOT)
		runs = [
			{**run, 'timestamp': ts, 'dir': f'{out_rel}/{ts}'}
			for ts, run in sorted(status.get('runs', {}).items(), reverse=True)
		]
		models = [
			{**m, 'hash': h, 'dir': f'{out_rel}/{m["class_name"]}_{h}'}
			for h, m in status.get('models', {}).items()
		]

		with open(cls.PATH, 'w') as f:
			f.write(cls._render(runs, models))
		return cls.PATH

	@staticmethod
	def _render(runs, models):
		data = json.dumps({'runs': runs, 'models': models})
		return f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>experiment dashboard</title>
<style>
	body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
	h1, h2 {{ font-weight: 600; }}
	table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
	th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; font-size: 14px; }}
	tbody tr.main-row:hover {{ background: #f7f7f7; cursor: pointer; }}
	.detail {{ background: #fafafa; padding: 1rem; border: 1px solid #eee; }}
	.detail img {{ max-width: 420px; margin: 4px; border: 1px solid #ccc; }}
	[hidden] {{ display: none; }}
	.pill {{ display: inline-block; padding: 2px 8px; border-radius: 10px; background: #eee; font-size: 12px; }}
</style>
</head>
<body>
<h1>experiment dashboard</h1>
<h2>runs</h2>
<table><tbody id="runs-body"></tbody></table>
<h2>models (by hash)</h2>
<table><tbody id="models-body"></tbody></table>
<script>
const DATA = {data};

function el(tag, attrs, children) {{
	const e = document.createElement(tag);
	for (const k in (attrs || {{}})) e.setAttribute(k, attrs[k]);
	(children || []).forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
	return e;
}}

function addExpandable(tbody, cols, cells, detailCols, detailChildren) {{
	const row = el('tr', {{class: 'main-row'}}, cells.map(c => el('td', {{}}, [String(c)])));
	const detail = el('div', {{class: 'detail', hidden: 'hidden'}}, detailChildren);
	const detailRow = el('tr', {{}}, [el('td', {{colspan: String(detailCols)}}, [detail])]);
	row.addEventListener('click', () => {{ detail.hidden = !detail.hidden; }});
	tbody.appendChild(row);
	tbody.appendChild(detailRow);
}}

function renderRuns() {{
	const tbody = document.getElementById('runs-body');
	tbody.appendChild(el('tr', {{}}, ['timestamp', 'stage', 'models'].map(h => el('th', {{}}, [h]))));
	DATA.runs.forEach(run => {{
		const imgs = ['training_curves', 'summary_table', 'final_metrics_bar', 'loss_distribution_over_epochs', 'qualitative_grid']
			.map(name => el('img', {{src: run.dir + '/' + name + '.png'}}));
		addExpandable(tbody,
			3, [run.timestamp, run.stage || 'unknown', (run.models || []).length],
			3, [el('p', {{}}, ['models: ' + (run.models || []).join(', ')]), ...imgs]);
	}});
}}

function renderModels() {{
	const tbody = document.getElementById('models-body');
	tbody.appendChild(el('tr', {{}}, ['hash', 'class', 'max epoch', 'best bce'].map(h => el('th', {{}}, [h]))));
	DATA.models.forEach(m => {{
		const bestBce = m.best_bce != null ? m.best_bce.toFixed(4) : '-';
		const ep = m.max_epoch_trained;
		addExpandable(tbody,
			4, [m.hash, m.class_name, ep, bestBce],
			4, [
				el('p', {{}}, ['enc: ' + JSON.stringify(m.enc_channels) + '  dec: ' + JSON.stringify(m.dec_channels)]),
				el('p', {{}}, ['used in runs: ' + (m.runs || []).join(', ')]),
				el('img', {{src: m.dir + '/training_curve_ep' + ep + '.png'}}),
				el('img', {{src: m.dir + '/predict_grid_ep' + ep + '.png'}}),
			]);
	}});
}}

renderRuns();
renderModels();
</script>
</body>
</html>
'''
