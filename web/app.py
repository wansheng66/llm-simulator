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
import threading
from datetime import datetime

# 添加项目根目录到 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.task_manager import task_manager
from core.benchmark_protocol import load_benchmark_spec

app = Flask(__name__, static_folder='.')
CORS(app)

BENCHMARK_RELEASE_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'benchmark_releases')
TASKS_FILE = os.path.join(PROJECT_ROOT, 'data', 'tasks.json')
BENCHMARK_SPEC_FILE = os.path.join(
    PROJECT_ROOT, 'configs', 'benchmark_specs',
    'qwen3_32b_fixed_batch_tp4_v1.json')
BENCHMARK_RELEASE_PATTERN = os.path.join(
    BENCHMARK_RELEASE_DIR, '**', 'benchmark_release.json')


def _benchmark_release_reports():
    """Return frozen formal Benchmark releases, newest first."""
    reports = []
    for path in glob.glob(BENCHMARK_RELEASE_PATTERN, recursive=True):
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
            if payload.get('experiment_type') != 'relative_benchmark_release':
                continue
            if payload.get('status') != 'frozen':
                continue
            fairness = payload.get('fairness', {})
            if not fairness.get('ground_truth_accepted'):
                continue
            if not fairness.get('runtime_validation_accepted'):
                continue
            if not fairness.get('all_points_have_ground_truth'):
                continue
            reports.append((os.path.getmtime(path), path, payload))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(reports, key=lambda item: item[0], reverse=True)


def _runtime_prediction_summary():
    """Build chart/table rows exclusively from frozen Benchmark releases."""
    rows = {}
    for _, release_path, release in _benchmark_release_reports():
        comparison = release.get('comparison', {})
        results = release.get('results', {})
        points = release.get('points', [])
        for side in ('reference', 'candidate'):
            identity = comparison.get(side, {})
            hardware_id = identity.get('gpu_type')
            tp = identity.get('tp')
            if not hardware_id or tp is None:
                continue
            key = (str(hardware_id), int(tp))
            if key in rows:
                continue
            case_scores = {}
            stage_values = {'prefill': [], 'decode': []}
            throughput_key = f'{side}_throughput_tok_s'
            for point in points:
                stage = point.get('stage')
                if stage not in stage_values:
                    continue
                throughput = float(point.get(throughput_key, 0))
                case_scores[point['case']] = throughput
                if throughput > 0:
                    stage_values[stage].append(throughput)
            rows[key] = {
                'gpu': hardware_id,
                'tp': int(tp),
                'prefill_avg': (
                    sum(stage_values['prefill']) /
                    len(stage_values['prefill'])
                    if stage_values['prefill'] else 0),
                'decode_avg': (
                    sum(stage_values['decode']) /
                    len(stage_values['decode'])
                    if stage_values['decode'] else 0),
                'comparison_status': 'frozen',
                'data_source': 'formal_benchmark_release',
                'case_scores': case_scores,
                'color': _hardware_color(hardware_id),
                'validation': {
                    'passed': True,
                    'relative_mape_pct': results.get('relative_mape_pct'),
                    'prefill_score_error_pct': results.get(
                        'p_score_error_pct'),
                    'decode_score_error_pct': results.get(
                        'd_score_error_pct'),
                    'rank_agreement_ratio': results.get(
                        'rank_agreement_ratio'),
                },
                'release_name': release.get('name'),
                'source_release': os.path.relpath(
                    release_path, PROJECT_ROOT),
            }
    return list(rows.values())


def _invert_runtime_point(point):
    result = dict(point)
    speedup = float(point['speedup'])
    truth = float(point['ground_truth_speedup'])
    result.update({
        'candidate_throughput_tok_s':
            point['reference_throughput_tok_s'],
        'reference_throughput_tok_s':
            point['candidate_throughput_tok_s'],
        'speedup': 1.0 / speedup,
        'ground_truth_speedup': 1.0 / truth,
        'relative_error_pct': abs(truth / speedup - 1.0) * 100.0,
    })
    return result


def _runtime_validation_result(candidate_gpu, reference_gpu, tp):
    """Return a frozen formal Benchmark comparison in requested direction."""
    for _, path, release in _benchmark_release_reports():
        comparison = release.get('comparison', {})
        stored_candidate = comparison.get('candidate', {})
        stored_reference = comparison.get('reference', {})
        if (int(stored_candidate.get('tp', -1)) != tp or
                int(stored_reference.get('tp', -1)) != tp):
            continue
        same = (stored_candidate.get('gpu_type') == candidate_gpu and
                stored_reference.get('gpu_type') == reference_gpu)
        reverse = (stored_candidate.get('gpu_type') == reference_gpu and
                   stored_reference.get('gpu_type') == candidate_gpu)
        if not (same or reverse):
            continue
        released = release.get('results', {})
        result = {
            'experiment_type': 'relative_benchmark_release_view',
            'status': 'verified',
            'candidate': dict(stored_candidate),
            'reference': dict(stored_reference),
            'scores': {
                'prefill_speedup': released.get('predicted_p_score'),
                'decode_speedup': released.get('predicted_d_score'),
                'paired_prefill_cases': released.get(
                    'paired_prefill_cases'),
                'paired_decode_cases': released.get(
                    'paired_decode_cases'),
            },
            'validation': {
                'passed': True,
                'ground_truth_complete': True,
                'relative_mape_pct': released.get('relative_mape_pct'),
                'rank_agreement_ratio': released.get(
                    'rank_agreement_ratio'),
                'ground_truth_prefill_speedup': released.get(
                    'ground_truth_p_score'),
                'ground_truth_decode_speedup': released.get(
                    'ground_truth_d_score'),
                'prefill_score_error_pct': released.get(
                    'p_score_error_pct'),
                'decode_score_error_pct': released.get(
                    'd_score_error_pct'),
            },
            'points': json.loads(json.dumps(release.get('points', []))),
            'release': {
                'name': release.get('name'),
                'status': release.get('status'),
                'created_at_utc': release.get('created_at_utc'),
                'path': os.path.relpath(path, PROJECT_ROOT),
            },
        }
        if reverse:
            result['candidate'], result['reference'] = (
                result['reference'], result['candidate'])
            result['points'] = [
                _invert_runtime_point(point)
                for point in result.get('points', [])
            ]
            scores = result['scores']
            validation = result['validation']
            for key in ('prefill_speedup', 'decode_speedup'):
                scores[key] = 1.0 / float(scores[key])
            for key in ('ground_truth_prefill_speedup',
                        'ground_truth_decode_speedup'):
                validation[key] = 1.0 / float(validation[key])
            validation['prefill_score_error_pct'] = abs(
                scores['prefill_speedup'] /
                validation['ground_truth_prefill_speedup'] - 1.0) * 100.0
            validation['decode_score_error_pct'] = abs(
                scores['decode_speedup'] /
                validation['ground_truth_decode_speedup'] - 1.0) * 100.0
            errors = [float(point['relative_error_pct'])
                      for point in result['points']]
            validation['relative_mape_pct'] = (
                sum(errors) / len(errors) if errors else None)
        result['score_kind'] = 'frozen_relative_benchmark_release'
        result['status_explanation'] = (
            '正式冻结 Benchmark；Runtime 预测已通过独立 18 点严格离线'
            '固定 Batch Ground Truth 验证')
        return result
    return None


def _hardware_color(hardware_id):
    name = str(hardware_id).lower()
    if 'l20' in name or 'nvidia' in name:
        return '#76b900'
    if any(token in name for token in ('ascend', 'atlas', '昇腾')):
        return '#e60012'
    return '#4ecdc4'


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
    # Only frozen releases enter official charts and tables.
    return jsonify(_runtime_prediction_summary())


@app.route('/api/benchmark/spec')
def get_benchmark_spec():
    """Return the versioned model workload and fairness contract."""
    return jsonify(load_benchmark_spec(BENCHMARK_SPEC_FILE))


@app.route('/api/benchmark/options')
def get_benchmark_options():
    options = set()
    for _, _, payload in _benchmark_release_reports():
        comparison = payload.get('comparison', {})
        for side in ('reference', 'candidate'):
            identity = comparison.get(side, {})
            hardware_id = identity.get('gpu_type')
            tp = identity.get('tp')
            if hardware_id is not None and tp is not None:
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
    runtime_result = _runtime_validation_result(
        candidate_gpu, reference_gpu, tp)
    if runtime_result is None:
        return jsonify({
            'error': '未找到 status=frozen 的正式 Benchmark 发布结果'
        }), 404
    return jsonify(runtime_result)


@app.route('/api/report/<gpu_type>')
def get_report(gpu_type):
    """Legacy single-GPU reports are excluded from the formal UI."""
    return jsonify({
        'error': '旧版单硬件报告已停用；请读取正式 Benchmark release'
    }), 410


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
    print(f"📁 正式 Benchmark 发布目录: {BENCHMARK_RELEASE_DIR}")
    print("🔒 主页面仅展示 status=frozen 的 benchmark_release.json")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
