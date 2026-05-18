from flask import Flask, request, jsonify
from svr_model import SVRModel
from robust_svr import RobustSVRModel
from online_svr import OnlineSVRModel
import os

app = Flask(__name__)

models = {}
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "SVR API 服务运行正常"})

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "缺少训练数据 X 或 y"}), 400
        
        X = data['X']
        y = data['y']
        
        model_id = data.get('model_id', 'default')
        kernel = data.get('kernel', 'rbf')
        C = data.get('C', 1.0)
        epsilon = data.get('epsilon', 0.1)
        gamma = data.get('gamma', 'scale')
        test_size = data.get('test_size', 0.2)
        use_robust = data.get('use_robust', False)
        
        if model_id not in models:
            if use_robust:
                models[model_id] = RobustSVRModel()
            else:
                models[model_id] = SVRModel()
        
        if use_robust:
            use_nusvr = data.get('use_nusvr', False)
            nu = data.get('nu', 0.5)
            remove_outliers = data.get('remove_outliers', True)
            outlier_method = data.get('outlier_method', 'combined')
            use_robust_scaler = data.get('use_robust_scaler', True)
            max_sv_ratio = data.get('max_sv_ratio', None)
            auto_tune = data.get('auto_tune', False)
            
            result = models[model_id].train(
                X=X, y=y,
                kernel=kernel,
                C=C,
                epsilon=epsilon,
                gamma=gamma,
                use_nusvr=use_nusvr,
                nu=nu,
                test_size=test_size,
                remove_outliers=remove_outliers,
                outlier_method=outlier_method,
                use_robust_scaler=use_robust_scaler,
                max_sv_ratio=max_sv_ratio,
                auto_tune=auto_tune
            )
        else:
            result = models[model_id].train(
                X=X, y=y,
                kernel=kernel,
                C=C,
                epsilon=epsilon,
                gamma=gamma,
                test_size=test_size
            )
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "message": "模型训练完成",
            "metrics": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'X' not in data:
            return jsonify({"error": "缺少预测数据 X"}), 400
        
        model_id = data.get('model_id', 'default')
        
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在，请先训练或加载模型"}), 404
        
        predictions = models[model_id].predict(data['X'])
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "predictions": predictions
        })
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/<model_id>', methods=['GET'])
def get_model_info(model_id):
    try:
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        info = models[model_id].get_model_info()
        return jsonify({
            "success": True,
            "model_id": model_id,
            "info": info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/<model_id>/save', methods=['POST'])
def save_model(model_id):
    try:
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        filepath = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        message = models[model_id].save_model(filepath)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "message": message,
            "filepath": filepath
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/<model_id>/outliers', methods=['GET'])
def get_outliers(model_id):
    try:
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not hasattr(models[model_id], 'get_outlier_info'):
            return jsonify({"error": "此模型不支持异常值信息查询，请使用鲁棒SVR"}), 400
        
        outlier_info = models[model_id].get_outlier_info()
        return jsonify({
            "success": True,
            "model_id": model_id,
            "outlier_info": outlier_info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model/<model_id>/load', methods=['POST'])
def load_model(model_id):
    try:
        filepath = os.path.join(MODEL_DIR, f"{model_id}.pkl")
        
        if model_id not in models:
            data = request.get_json() or {}
            use_robust = data.get('use_robust', False)
            if use_robust:
                models[model_id] = RobustSVRModel()
            else:
                models[model_id] = SVRModel()
        
        message = models[model_id].load_model(filepath)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "message": message
        })
    
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models', methods=['GET'])
def list_models():
    try:
        model_list = []
        for model_id, model in models.items():
            model_list.append({
                "model_id": model_id,
                "info": model.get_model_info()
            })
        
        return jsonify({
            "success": True,
            "models": model_list
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/init', methods=['POST'])
def online_init():
    try:
        data = request.get_json()
        
        if 'X' not in data or 'y' not in data:
            return jsonify({"error": "缺少初始训练数据 X 或 y"}), 400
        
        model_id = data.get('model_id', 'online_default')
        kernel = data.get('kernel', 'rbf')
        C = data.get('C', 1.0)
        epsilon = data.get('epsilon', 0.1)
        gamma = data.get('gamma', 'scale')
        retrain_threshold = data.get('retrain_threshold', 50)
        new_sample_weight = data.get('new_sample_weight', 2.0)
        preserve_support_vectors = data.get('preserve_support_vectors', True)
        
        if model_id not in models:
            models[model_id] = OnlineSVRModel(
                retrain_threshold=retrain_threshold,
                new_sample_weight=new_sample_weight,
                preserve_support_vectors=preserve_support_vectors
            )
        elif not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        result = models[model_id].initial_train(
            X=data['X'],
            y=data['y'],
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma
        )
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "message": "在线SVR模型初始化完成",
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/add', methods=['POST'])
def online_add_sample():
    try:
        data = request.get_json()
        
        if 'X' not in data:
            return jsonify({"error": "缺少样本数据 X"}), 400
        
        model_id = data.get('model_id', 'online_default')
        auto_retrain = data.get('auto_retrain', True)
        y = data.get('y', None)
        
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在，请先初始化"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        result = models[model_id].add_sample(
            X=data['X'],
            y=y,
            auto_retrain=auto_retrain
        )
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/retrain', methods=['POST'])
def online_retrain():
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id', 'online_default')
        warm_start = data.get('warm_start', True)
        
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        result = models[model_id].retrain(reason='manual', warm_start=warm_start)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/predict', methods=['POST'])
def online_predict():
    try:
        data = request.get_json()
        
        if 'X' not in data:
            return jsonify({"error": "缺少预测数据 X"}), 400
        
        model_id = data.get('model_id', 'online_default')
        with_confidence = data.get('with_confidence', False)
        
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        if with_confidence:
            predictions = models[model_id].predict_with_confidence(data['X'])
            predictions = [{'prediction': p, 'confidence': c} for p, c in predictions]
        else:
            predictions = models[model_id].predict(data['X'])
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "predictions": predictions,
            "with_confidence": with_confidence
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/model/<model_id>/buffer', methods=['GET'])
def online_get_buffer(model_id):
    try:
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        buffer_info = models[model_id].get_buffer_info()
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "buffer_info": buffer_info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/model/<model_id>/buffer', methods=['DELETE'])
def online_clear_buffer(model_id):
    try:
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        result = models[model_id].clear_buffer()
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/online/model/<model_id>/params', methods=['PUT'])
def online_update_params(model_id):
    try:
        data = request.get_json()
        
        if model_id not in models:
            return jsonify({"error": f"模型 {model_id} 不存在"}), 404
        
        if not isinstance(models[model_id], OnlineSVRModel):
            return jsonify({"error": f"模型 {model_id} 不是在线SVR模型"}), 400
        
        result = models[model_id].update_params(**data)
        
        return jsonify({
            "success": True,
            "model_id": model_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("SVR API 服务启动中...")
    print("=" * 70)
    print("基础接口:")
    print("  GET  /health                   - 健康检查")
    print("  POST /api/train                - 训练模型 (use_robust=True 启用鲁棒模式)")
    print("  POST /api/predict              - 预测")
    print("  GET  /api/model/<id>           - 获取模型信息")
    print("  GET  /api/model/<id>/outliers - 获取异常值信息")
    print("  POST /api/model/<id>/save      - 保存模型")
    print("  POST /api/model/<id>/load      - 加载模型")
    print("  GET  /api/models               - 列出所有模型")
    print("-" * 70)
    print("在线学习接口 (Online SVR):")
    print("  POST /api/online/init          - 初始化在线SVR模型")
    print("  POST /api/online/add           - 增量添加样本")
    print("  POST /api/online/retrain       - 手动触发重训练")
    print("  POST /api/online/predict       - 在线模型预测")
    print("  GET  /api/online/model/<id>/buffer - 获取缓存信息")
    print("  DELETE /api/online/model/<id>/buffer - 清空缓存")
    print("  PUT  /api/online/model/<id>/params - 更新模型参数")
    print("=" * 70)
    print("关键特性:")
    print("  ✓ 支持向量保留机制 - 避免完全重训练")
    print("  ✓ 智能重训练触发 - 基于样本数/时间")
    print("  ✓ 新样本加权 - 让新数据影响更大")
    print("  ✓ 预测置信度 - 基于历史数据距离")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)
