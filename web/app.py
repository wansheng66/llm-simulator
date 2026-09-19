#!/usr/bin/env python3
"""
Benchmark 平台后端 API
启动: python web/app.py
访问: http://localhost:5000
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import glob
import sys
import csv
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.task_manager import task_manager
from core.benchmark_protocol import (
    adapt_legacy_report,
    compare_reports,
    load_benchmark_spec,
)

app = Flask(__name__, static_folder='.')
CORS(app)

REPORT_DIR = os.path.join(PROJECT_ROOT, 'data', 'benchmark_reports')
RELATIVE_REPORT_DIR = os.path.join(PROJECT_ROOT, 'data', 'relative_benchmark')
TASKS_FILE = os.path.join(PROJECT_ROOT, 'data', 'tasks.json')
BENCHMARK_SPEC_FILE = os.path.join(
    PROJECT_ROOT, 'configs', 'benchmark_specs',
    'qwen3_32b_fixed_batch_tp4_v1.json')
FIXED_BATCH_RELATIVE_PATTERN = os.path.join(
    PROJECT_ROOT, 'data', 'relative_benchmark', '**',
    'fixed_batch_relative_report.json')


def _latest_report(gpu_type, tp):
    pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu_type}_TP{tp}_*.json')
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def _load_comparable_report(path):
    """Load a report and explicitly adapt legacy v0 files in memory.

    Legacy files used seq/time and 1/time.  Recompute token throughput from
    batch and total time, and mark the provenance so the UI cannot present the
    result as independently verified ground truth.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    spec = load_benchmark_spec(BENCHMARK_SPEC_FILE)
    return adapt_legacy_report(payload, spec)


def _ground_truth_ratios(candidate_gpu, reference_gpu, tp):
    pattern = os.path.join(
        PROJECT_ROOT, 'data', 'relative_ground_truth', '**',
        'relative_ground_truth_report.json')
    matches = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            acceptance = payload.get('acceptance', {})
            if not (acceptance.get('ground_truth_complete') and
                    acceptance.get('runtime_compatible') and
                    acceptance.get('repeatability_ok')):
                continue
            candidate = payload.get('candidate', {})
            reference = payload.get('reference', {})
            same = (candidate.get('hardware_id') == candidate_gpu and
                    reference.get('hardware_id') == reference_gpu and
                    int(candidate.get('tp_size', -1)) == tp and
                    int(reference.get('tp_size', -1)) == tp)
            reverse = (candidate.get('hardware_id') == reference_gpu and
                       reference.get('hardware_id') == candidate_gpu and
                       int(candidate.get('tp_size', -1)) == tp and
                       int(reference.get('tp_size', -1)) == tp)
            if same:
                matches.append((os.path.getmtime(path), path,
                                payload.get('ground_truth_ratios', {})))
            elif reverse:
                ratios = {key: 1.0 / float(value)
                          for key, value in payload.get('ground_truth_ratios', {}).items()
                          if float(value) > 0}
                matches.append((os.path.getmtime(path), path, ratios))
        except (OSError, ValueError, TypeError, ZeroDivisionError):
            continue
    if not matches:
        return None, None
    _, path, ratios = max(matches, key=lambda item: item[0])
    return ratios, path


def _fixed_batch_relative_reports():
    """Return accepted real fixed-batch A/B reports, newest first."""
    reports = []
    for path in glob.glob(FIXED_BATCH_RELATIVE_PATTERN, recursive=True):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            if payload.get('experiment_type') != 'fixed_batch_relative_ground_truth':
                continue
            acceptance = payload.get('acceptance', {})
            if not (acceptance.get('passed') and
                    acceptance.get('all_points_repeatable') and
                    int(acceptance.get('paired_point_count', 0)) == 18):
                continue
            reports.append((os.path.getmtime(path), path, payload))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(reports, key=lambda item: item[0], reverse=True)


def _hardware_color(hardware_id):
    name = str(hardware_id).lower()
    if 'l20' in name or 'nvidia' in name:
        return '#76b900'
    if any(token in name for token in ('ascend', 'atlas', '昇腾')):
        return '#e60012'
    return '#4ecdc4'


def _fixed_batch_summary():
    """Build chart/table rows directly from accepted fixed-batch truth."""
    rows = {}
    for _, path, payload in _fixed_batch_relative_reports():
        tp = int(payload.get('identity', {}).get('tp_size', -1))
        if tp <= 0:
            continue
        sides = {
            'reference': {
                'hardware_id': payload.get('reference', {}).get('hardware_id'),
                'throughput_key': 'reference_throughput_tokens_per_s',
                'cv_key': 'reference_cv_pct',
            },
            'candidate': {
                'hardware_id': payload.get('candidate', {}).get('hardware_id'),
                'throughput_key': 'candidate_throughput_tokens_per_s',
                'cv_key': 'candidate_cv_pct',
            },
        }
        for side in sides.values():
            hardware_id = side['hardware_id']
            key = (hardware_id, tp)
            if not hardware_id or key in rows:
                continue
            case_scores = {}
            prefill_values = []
            decode_values = []
            cv_values = []
            for point in payload.get('points', []):
                stage = point.get('stage')
                batch = int(point.get('batch_size', 0))
                length = int(point.get('representative_length', 0))
                prefix = 'P' if stage == 'prefill' else 'D'
                length_name = 'L' if stage == 'prefill' else 'KV'
                case_name = f'{prefix}_B{batch}_{length_name}{length}'
                throughput = float(point.get(side['throughput_key'], 0))
                case_scores[case_name] = throughput
                cv_values.append(float(point.get(side['cv_key'], 0)))
                if stage == 'prefill':
                    prefill_values.append(throughput)
                elif stage == 'decode':
                    decode_values.append(throughput)
            rows[key] = {
                'gpu': hardware_id,
                'tp': tp,
                'prefill_avg': (sum(prefill_values) / len(prefill_values)
                                if prefill_values else 0),
                'decode_avg': (sum(decode_values) / len(decode_values)
                               if decode_values else 0),
                'combined_score': None,
                'combined_score_status': 'undefined_without_pd_mix',
                'comparison_status': 'verified',
                'data_source': 'fixed_batch_ground_truth',
                'max_cv_pct': max(cv_values) if cv_values else None,
                'source_report': os.path.relpath(path, PROJECT_ROOT),
                'price': None,
                'price_performance': None,
                'case_scores': case_scores,
                'color': _hardware_color(hardware_id),
            }
    return list(rows.values())


def _invert_fixed_batch_point(point):
    speedup = float(point['candidate_speedup_vs_reference'])
    if speedup <= 0:
        raise ValueError('fixed-batch speedup must be positive')
    winner = point.get('winner')
    inverse_winner = {
        'candidate': 'reference',
        'reference': 'candidate',
        'tie': 'tie',
    }.get(winner, winner)
    result = dict(point)
    result.update({
        'reference_mean_ms': point['candidate_mean_ms'],
        'candidate_mean_ms': point['reference_mean_ms'],
        'reference_throughput_tokens_per_s':
            point['candidate_throughput_tokens_per_s'],
        'candidate_throughput_tokens_per_s':
            point['reference_throughput_tokens_per_s'],
        'candidate_speedup_vs_reference': 1.0 / speedup,
        'winner': inverse_winner,
        'reference_cv_pct': point['candidate_cv_pct'],
        'candidate_cv_pct': point['reference_cv_pct'],
    })
    return result


def _fixed_batch_relative_result(candidate_gpu, reference_gpu, tp):
    """Find real strict fixed-batch truth and orient it as candidate/reference."""
    for _, path, payload in _fixed_batch_relative_reports():
        if int(payload.get('identity', {}).get('tp_size', -1)) != tp:
            continue
        stored_candidate = payload.get('candidate', {}).get('hardware_id')
        stored_reference = payload.get('reference', {}).get('hardware_id')
        same = (stored_candidate == candidate_gpu and
                stored_reference == reference_gpu)
        reverse = (stored_candidate == reference_gpu and
                   stored_reference == candidate_gpu)
        if not (same or reverse):
            continue

        scores = payload.get('scores', {})
        p_score = float(scores['p_score'])
        d_score = float(scores['d_score'])
        points = payload.get('points', [])
        winner_counts = payload.get('winner_counts', {})
        if reverse:
            p_score = 1.0 / p_score
            d_score = 1.0 / d_score
            points = [_invert_fixed_batch_point(point) for point in points]
            winner_counts = {
                stage: {
                    'candidate': counts.get('reference', 0),
                    'reference': counts.get('candidate', 0),
                    'tie': counts.get('tie', 0),
                }
                for stage, counts in winner_counts.items()
            }

        prefill_count = sum(point.get('stage') == 'prefill' for point in points)
        decode_count = sum(point.get('stage') == 'decode' for point in points)
        candidate_manifest = payload.get(
            'reference' if reverse else 'candidate', {}).get('manifest')
        reference_manifest = payload.get(
            'candidate' if reverse else 'reference', {}).get('manifest')
        return {
            'schema_version': 1,
            'status': 'verified',
            'score_kind': 'measured_ground_truth',
            'status_explanation': (
                '严格离线固定 Batch 实测；同模型、同 TP、同负载、同测量协议'),
            'candidate': {'gpu_type': candidate_gpu, 'tp': tp},
            'reference': {'gpu_type': reference_gpu, 'tp': tp},
            'scores': {
                'prefill_speedup': p_score,
                'decode_speedup': d_score,
                'paired_prefill_cases': prefill_count,
                'paired_decode_cases': decode_count,
            },
            'validation': {
                'ground_truth_complete': True,
                'repeatability_ok': True,
                'max_cv_pct': payload.get('acceptance', {}).get('max_cv_pct'),
            },
            'winner_counts': winner_counts,
            'points': points,
            'sources': {
                'ground_truth': os.path.relpath(path, PROJECT_ROOT),
                'candidate': candidate_manifest,
                'reference': reference_manifest,
            },
        }
    return None


def _get_task_from_file(task_id):
    """直接从文件读取任务，确保实时性"""
    if not os.path.exists(TASKS_FILE):
        return None
    try:
        with open(TASKS_FILE, 'r') as f:
            tasks = json.load(f)
        for task in tasks:
            if task.get('id') == task_id:
                return task
    except Exception as e:
        print(f"读取任务文件失败: {e}")
    return None

def _get_task_logs_from_file(task_id, since=0):
    """直接从文件读取日志"""
    task = _get_task_from_file(task_id)
    if not task:
        return []
    logs = task.get('logs', [])
    return logs[since:]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/summary')
def get_summary():
    results = _fixed_batch_summary()
    seen = {(item['gpu'], int(item['tp'])) for item in results}
    summary_csv = os.path.join(REPORT_DIR, 'benchmark_summary.csv')
    if os.path.exists(summary_csv):
        with open(summary_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gpu = row['GPU']
                tp = int(row['TP']) if row['TP'] else 1
                if (gpu, tp) in seen:
                    continue
                
                # 构建 case_scores：从该 GPU 的所有 JSON 文件中读取数据
                case_scores = {}
                pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu}_TP{tp}_*.json')
                files = glob.glob(pattern)
                data = None
                if files:
                    # 读取匹配的 JSON 文件（按 TP 精确匹配）
                    latest = max(files, key=os.path.getmtime)
                    data = _load_comparable_report(latest)
                    for item in data.get('prefill', []):
                        case_scores[item['case']] = item.get('throughput_tok_s', 0)
                    for item in data.get('decode', []):
                        case_scores[item['case']] = item.get('throughput_tok_s', 0)
                prefill_values = [item.get('throughput_tok_s', 0)
                                  for item in (data or {}).get('prefill', [])]
                decode_values = [item.get('throughput_tok_s', 0)
                                 for item in (data or {}).get('decode', [])]
                
                results.append({
                    'gpu': gpu,
                    'tp': tp,
                    'prefill_avg': (sum(prefill_values) / len(prefill_values)
                                    if prefill_values else 0),
                    'decode_avg': (sum(decode_values) / len(decode_values)
                                   if decode_values else 0),
                    'combined_score': float(row['综合评分']) if row['综合评分'] else 0,
                    'combined_score_status': 'legacy_not_for_ranking',
                    'comparison_status': 'provisional',
                    'price': float(row['价格']) if row.get('价格') and row['价格'] else None,
                    'price_performance': float(row['性价比']) if row.get('性价比') and row['性价比'] else None,
                    'case_scores': case_scores,
                    'data_source': 'legacy_benchmark_report',
                    'color': _hardware_color(gpu),
                })
        return jsonify(results)
    
    # fallback: 读取所有 JSON 报告
    json_files = glob.glob(os.path.join(REPORT_DIR, 'benchmark_*.json'))
    for f in json_files:
        data = _load_comparable_report(f)
        meta = data.get('meta', {})
        summary = data.get('summary', {})
        gpu = meta.get('gpu_type', 'Unknown')
        tp = meta.get('tp', 1)
        if (gpu, int(tp)) in seen:
            continue
        case_scores = {}
        for item in data.get('prefill', []):
            case_scores[item['case']] = item.get('throughput_tok_s', 0)
        for item in data.get('decode', []):
            case_scores[item['case']] = item.get('throughput_tok_s', 0)
        prefill_values = [item.get('throughput_tok_s', 0)
                          for item in data.get('prefill', [])]
        decode_values = [item.get('throughput_tok_s', 0)
                         for item in data.get('decode', [])]
        results.append({
            'gpu': gpu,
            'tp': tp,
            'prefill_avg': (sum(prefill_values) / len(prefill_values)
                            if prefill_values else 0),
            'decode_avg': (sum(decode_values) / len(decode_values)
                           if decode_values else 0),
            'combined_score': summary.get('combined_score', 0),
            'combined_score_status': 'legacy_not_for_ranking',
            'comparison_status': 'provisional',
            'price': meta.get('price'),
            'case_scores': case_scores,
            'data_source': 'legacy_benchmark_report',
            'color': _hardware_color(gpu),
        })
    return jsonify(results)


@app.route('/api/benchmark/spec')
def get_benchmark_spec():
    """Return the versioned model workload and fairness contract."""
    return jsonify(load_benchmark_spec(BENCHMARK_SPEC_FILE))


@app.route('/api/benchmark/options')
def get_benchmark_options():
    options = set()
    for path in glob.glob(os.path.join(REPORT_DIR, 'benchmark_*_TP*_*.json')):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                meta = json.load(handle).get('meta', {})
            if meta.get('gpu_type') and meta.get('tp'):
                options.add((str(meta['gpu_type']), int(meta['tp'])))
        except (OSError, ValueError, TypeError):
            continue
    for _, _, payload in _fixed_batch_relative_reports():
        tp = payload.get('identity', {}).get('tp_size')
        for side in ('reference', 'candidate'):
            hardware_id = payload.get(side, {}).get('hardware_id')
            if hardware_id and tp is not None:
                options.add((str(hardware_id), int(tp)))
    return jsonify([
        {'gpu': gpu, 'tp': tp, 'id': f'{gpu}::TP{tp}'}
        for gpu, tp in sorted(options)
    ])


@app.route('/api/benchmark/relative')
def get_relative_benchmark():
    candidate_gpu = request.args.get('candidate')
    reference_gpu = request.args.get('reference')
    tp = request.args.get('tp', type=int)
    if not candidate_gpu or not reference_gpu or tp is None:
        return jsonify({'error': 'candidate, reference and tp are required'}), 400
    fixed_batch_truth = _fixed_batch_relative_result(
        candidate_gpu, reference_gpu, tp)
    if fixed_batch_truth is not None:
        return jsonify(fixed_batch_truth)
    candidate_path = _latest_report(candidate_gpu, tp)
    reference_path = _latest_report(reference_gpu, tp)
    if not candidate_path or not reference_path:
        return jsonify({'error': 'matching report not found'}), 404
    candidate = _load_comparable_report(candidate_path)
    reference = _load_comparable_report(reference_path)
    truth_ratios, truth_path = _ground_truth_ratios(
        candidate_gpu, reference_gpu, tp)
    result = compare_reports(candidate, reference, truth_ratios)
    result['sources'] = {
        'candidate': os.path.basename(candidate_path),
        'reference': os.path.basename(reference_path),
        'ground_truth': (os.path.relpath(truth_path, PROJECT_ROOT)
                         if truth_path else None),
    }
    if any(report.get('meta', {}).get('report_provenance') ==
           'legacy_v0_adapted_in_memory' for report in (candidate, reference)):
        result['compatibility']['warnings'].append(
            'legacy report throughput was corrected in memory; regenerate reports for protocol-native artifacts')
    return jsonify(result)


@app.route('/api/report/<gpu_type>')
def get_report(gpu_type):
    """获取某个 GPU 的详细报告"""
    pattern = os.path.join(REPORT_DIR, f'benchmark_{gpu_type}_*.json')
    files = glob.glob(pattern)
    if not files:
        return jsonify({'error': 'Not found'}), 404
    with open(files[-1], 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务列表"""
    limit = request.args.get('limit', 50, type=int)
    tasks = task_manager.get_tasks(limit)
    return jsonify(tasks)


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    task = _get_task_from_file(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


@app.route('/api/tasks/<task_id>/logs', methods=['GET'])
def get_task_logs(task_id):
    since = request.args.get('since', 0, type=int)
    logs = _get_task_logs_from_file(task_id, since)
    return jsonify({'logs': logs, 'count': len(logs)})


@app.route('/api/profile/start', methods=['POST'])
def start_profile():
    import sys
    from datetime import datetime
    import uuid
    import json
    import tempfile
    from core.task_manager import task_manager

    data = request.get_json()
    
    # ========== 1. 基础字段校验 ==========
    required = ['host', 'user', 'gpu_type']
    for key in required:
        if key not in data or not data[key]:
            return jsonify({'error': f'Missing or empty required field: {key}'}), 400
    
    # ========== 2. TP 校验 ==========
    tps = data.get('tps', [])
    if not tps:
        if data.get('tp'):
            tps = [data.get('tp')]
        else:
            return jsonify({'error': '至少选择一个 TP 配置'}), 400
    
    try:
        tps = sorted(set(int(t) for t in tps))
    except (ValueError, TypeError):
        return jsonify({'error': 'TP 值必须为数字'}), 400
    
    valid_tps = [1, 2, 4, 8, 16]
    for tp in tps:
        if tp not in valid_tps:
            return jsonify({'error': f'TP={tp} 不在支持的范围内: {valid_tps}'}), 400
    
    # ========== 3. 其他字段处理 ==========
    gpu_type = data['gpu_type'].strip()
    if not gpu_type:
        return jsonify({'error': 'GPU 型号不能为空'}), 400
    
    # ========== 新增：获取跳过层时间选项 ==========
    skip_layer_bench = data.get('skip_layer_bench', True)  # 默认跳过
    
    # ========== 4. 生成真实任务 ID ==========
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    # ========== 5. 创建任务对象 ==========
    task = {
        'id': task_id,
        'type': 'profile_batch',
        'status': 'pending',
        'params': {
            'gpu_type': gpu_type,
            'host': data['host'].strip(),
            'user': data['user'].strip(),
            'password': data.get('password'),
            'port': data.get('port', 22),
            'price': data.get('price'),
            'tps': tps,
            'total': len(tps),
            'skip_layer_bench': skip_layer_bench  # 新增
        },
        'created_at': datetime.now().isoformat(),
        'started_at': None,
        'completed_at': None,
        'logs': [],
        'result': None,
        'error': None
    }
    
    # ========== 6. 原子写入 tasks.json ==========
    try:
        if os.path.exists(TASKS_FILE):
            with open(TASKS_FILE, 'r') as f:
                tasks = json.load(f)
        else:
            tasks = []
        tasks.append(task)
        
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(TASKS_FILE), suffix='.json')
        with os.fdopen(fd, 'w') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, TASKS_FILE)
        
        sys.stderr.write(f"[任务创建] task_id={task_id}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[任务创建失败] {e}\n")
        sys.stderr.flush()
        return jsonify({'error': f'任务创建失败: {str(e)}'}), 500
    
    # ========== 7. 关键修复：同步更新 task_manager 内存缓存 ==========
    with task_manager._lock:
        task_manager.tasks.append(task)  # 确保 _run_task 能找到任务
    
    # ========== 8. 异步执行任务 ==========
    def background_executor():
        import sys
        sys.stderr.write("🔥🔥🔥 background_executor 线程已启动\n")
        sys.stderr.flush()
        try:
            task_manager._run_task(task_id)
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[后台执行失败] {e}\n")
            sys.stderr.flush()

    import sys
    sys.stderr.write(f"准备启动线程，task_id={task_id}\n")
    sys.stderr.flush()
    thread = threading.Thread(target=background_executor, daemon=True)
    thread.start()
    sys.stderr.write(f"✅ 线程已启动，task_id={task_id}\n")
    sys.stderr.flush()
    
    # ========== 9. 立即返回 ==========
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': f'已提交 {len(tps)} 个 TP 配置的采集任务' +
                   (' (跳过层时间)' if skip_layer_bench else ' (包含层时间)')
    })

@app.route('/api/benchmark/run', methods=['POST'])
def run_benchmark():
    """仅运行评分引擎（不采集新数据）"""
    data = request.get_json()
    
    if 'gpu_type' not in data:
        return jsonify({'error': 'Missing gpu_type'}), 400
    
    task_id = task_manager.submit_task('benchmark', {
        'gpu_type': data['gpu_type'],
        'tp': data.get('tp', 1),
        'price': data.get('price')
    })
    
    return jsonify({
        'task_id': task_id,
        'status': 'pending',
        'message': f'Benchmark task {task_id} submitted'
    })


@app.route('/api/gpu-types')
def get_gpu_types():
    """获取所有已采集的 GPU 类型"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    gpu_dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item not in ['benchmark_reports', 'ground_truth_pd', 'trace']:
            # 检查是否包含 Profiling 数据
            has_data = any(f.endswith('.json') for f in os.listdir(item_path))
            if has_data:
                gpu_dirs.append(item)
    return jsonify(gpu_dirs)


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Benchmark 平台启动")
    print("=" * 60)
    print(f"📊 访问: http://localhost:5000")
    print(f"📁 传统报告目录: {REPORT_DIR}")
    print(f"📁 相对 Benchmark 目录: {RELATIVE_REPORT_DIR}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
