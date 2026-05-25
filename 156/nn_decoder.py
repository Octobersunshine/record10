import numpy as np
import pickle
import os
from collections import defaultdict


class NeuralDecoder:
    def __init__(self, distance, hidden_dim=64, num_layers=3):
        self.d = distance
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.n_x_stabs = distance * (distance - 1)
        self.n_z_stabs = (distance - 1) * distance
        self.n_data = distance * distance
        
        self._build_graph_structure()
        self._initialize_weights()
        
        self.trained = False
    
    def _build_graph_structure(self):
        d = self.d
        self.stab_neighbors = defaultdict(list)
        self.stab_to_qubit = {}
        
        for i in range(d):
            for j in range(d - 1):
                stab_idx = i * (d - 1) + j
                qubits = []
                qubits.append(i * d + j)
                qubits.append(i * d + j + 1)
                if i + 1 < d:
                    qubits.append((i + 1) * d + j)
                    qubits.append((i + 1) * d + j + 1)
                self.stab_to_qubit[('x', stab_idx)] = qubits
                
                for q in qubits:
                    self.stab_neighbors[('x', stab_idx)].append(q)
        
        for i in range(d - 1):
            for j in range(d):
                stab_idx = i * d + j
                qubits = []
                qubits.append(i * d + j)
                qubits.append((i + 1) * d + j)
                if j + 1 < d:
                    qubits.append(i * d + j + 1)
                    qubits.append((i + 1) * d + j + 1)
                self.stab_to_qubit[('z', stab_idx)] = qubits
                
                for q in qubits:
                    self.stab_neighbors[('z', stab_idx)].append(q)
    
    def _initialize_weights(self):
        np.random.seed(42)
        
        input_dim = 2
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for _ in range(self.num_layers):
            W = np.random.randn(prev_dim, self.hidden_dim) * 0.1
            b = np.zeros(self.hidden_dim)
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = self.hidden_dim
        
        self.output_W = np.random.randn(self.hidden_dim, 1) * 0.1
        self.output_b = np.zeros(1)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def _prepare_node_features(self, x_syndrome, z_syndrome):
        n_nodes = self.n_x_stabs + self.n_z_stabs + self.n_data
        
        features = np.zeros((n_nodes, 2))
        
        for i in range(self.n_x_stabs):
            features[i, 0] = x_syndrome[i]
        
        offset = self.n_x_stabs
        for i in range(self.n_z_stabs):
            features[offset + i, 1] = z_syndrome[i]
        
        return features
    
    def _message_passing(self, features):
        for layer in range(self.num_layers):
            new_features = np.copy(features)
            
            for stab_type in ['x', 'z']:
                n_stabs = self.n_x_stabs if stab_type == 'x' else self.n_z_stabs
                offset = 0 if stab_type == 'x' else self.n_x_stabs
                
                for stab_idx in range(n_stabs):
                    node_idx = offset + stab_idx
                    neighbors = self.stab_neighbors.get((stab_type, stab_idx), [])
                    
                    if neighbors:
                        neighbor_offset = self.n_x_stabs + self.n_z_stabs
                        neighbor_feats = features[neighbor_offset + np.array(neighbors)]
                        aggregated = np.mean(neighbor_feats, axis=0)
                        new_features[node_idx] = 0.5 * features[node_idx] + 0.5 * aggregated
            
            features = new_features @ self.weights[layer] + self.biases[layer]
            features = self._relu(features)
        
        return features
    
    def _predict_corrections(self, features):
        output = features @ self.output_W + self.output_b
        output = self._sigmoid(output)
        return output.flatten()
    
    def decode(self, x_syndrome, z_syndrome):
        features = self._prepare_node_features(x_syndrome, z_syndrome)
        features = self._message_passing(features)
        predictions = self._predict_corrections(features)
        
        qubit_offset = self.n_x_stabs + self.n_z_stabs
        qubit_predictions = predictions[qubit_offset:qubit_offset + self.n_data]
        
        x_correction = qubit_predictions > 0.5
        z_correction = np.zeros_like(x_correction)
        
        return x_correction, z_correction
    
    def apply_correction(self, surface_code, x_correction, z_correction):
        for i in range(self.n_data):
            if x_correction[i]:
                surface_code.apply_bit_flip(i)
            if z_correction[i]:
                surface_code.apply_phase_flip(i)


class GNNDecoderTrainer:
    def __init__(self, distance, hidden_dim=64, num_layers=3, lr=0.001):
        self.decoder = NeuralDecoder(distance, hidden_dim, num_layers)
        self.lr = lr
        self.loss_history = []
    
    def generate_training_sample(self, p_error=0.05):
        from surface_code import SurfaceCode
        from error_model import ErrorModel
        
        sc = SurfaceCode(self.decoder.d)
        error_model = ErrorModel(p_bit_flip=p_error, p_phase_flip=p_error)
        
        error_model.apply_errors(sc)
        x_errors = sc.x_errors.copy()
        z_errors = sc.z_errors.copy()
        
        x_syndrome, z_syndrome = sc.measure_stabilizers()
        
        return x_syndrome, z_syndrome, x_errors, z_errors
    
    def train_step(self, x_syndrome, z_syndrome, x_target, z_target):
        features = self.decoder._prepare_node_features(x_syndrome, z_syndrome)
        features = self.decoder._message_passing(features)
        predictions = self.decoder._predict_corrections(features)
        
        qubit_offset = self.decoder.n_x_stabs + self.decoder.n_z_stabs
        
        target = np.zeros(self.decoder.n_data)
        for i in range(self.decoder.n_data):
            if x_target[i] or z_target[i]:
                target[i] = 1.0
        
        pred = predictions[qubit_offset:qubit_offset + self.decoder.n_data]
        
        loss = -np.mean(target * np.log(pred + 1e-8) + (1 - target) * np.log(1 - pred + 1e-8))
        
        d_pred = pred - target
        
        d_output = np.zeros_like(predictions)
        d_output[qubit_offset:qubit_offset + self.decoder.n_data] = d_pred / self.decoder.n_data
        
        d_output_W = features.T @ d_output.reshape(-1, 1)
        d_output_b = np.sum(d_output)
        
        self.decoder.output_W -= self.lr * d_output_W
        self.decoder.output_b -= self.lr * d_output_b
        
        return loss
    
    def train(self, n_samples=10000, p_error=0.05, batch_size=32, verbose=True):
        losses = []
        
        for i in range(n_samples):
            x_syn, z_syn, x_tar, z_tar = self.generate_training_sample(p_error)
            loss = self.train_step(x_syn, z_syn, x_tar, z_tar)
            losses.append(loss)
            
            if verbose and (i + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"Sample {i+1}/{n_samples}, Avg Loss: {avg_loss:.6f}")
        
        self.loss_history.extend(losses)
        self.decoder.trained = True
        return losses
    
    def save_model(self, filepath):
        model_data = {
            'd': self.decoder.d,
            'hidden_dim': self.decoder.hidden_dim,
            'num_layers': self.decoder.num_layers,
            'weights': self.decoder.weights,
            'biases': self.decoder.biases,
            'output_W': self.decoder.output_W,
            'output_b': self.decoder.output_b,
            'loss_history': self.loss_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.decoder.d = model_data['d']
        self.decoder.hidden_dim = model_data['hidden_dim']
        self.decoder.num_layers = model_data['num_layers']
        self.decoder.weights = model_data['weights']
        self.decoder.biases = model_data['biases']
        self.decoder.output_W = model_data['output_W']
        self.decoder.output_b = model_data['output_b']
        self.loss_history = model_data['loss_history']
        self.decoder.trained = True


class FastNeuralDecoder:
    def __init__(self, distance):
        self.d = distance
        self.n_x_stabs = distance * (distance - 1)
        self.n_z_stabs = (distance - 1) * distance
        self.n_data = distance * distance
        
        self.x_classifier = None
        self.z_classifier = None
        self.trained = False
    
    def _extract_features(self, syndrome, stab_type):
        features = syndrome.astype(np.float32)
        return features
    
    def train(self, n_samples=5000, p_error=0.05, verbose=True):
        from surface_code import SurfaceCode
        from error_model import ErrorModel
        
        X_x_train = []
        X_z_train = []
        y_x_train = []
        y_z_train = []
        
        for _ in range(n_samples):
            sc = SurfaceCode(self.d)
            error_model = ErrorModel(p_bit_flip=p_error, p_phase_flip=p_error)
            error_model.apply_errors(sc)
            
            x_errors = sc.x_errors.copy()
            z_errors = sc.z_errors.copy()
            
            x_syn, z_syn = sc.measure_stabilizers()
            
            X_x_train.append(x_syn.astype(np.float32))
            X_z_train.append(z_syn.astype(np.float32))
            y_x_train.append(x_errors.astype(np.float32))
            y_z_train.append(z_errors.astype(np.float32))
        
        X_x_train = np.array(X_x_train)
        X_z_train = np.array(X_z_train)
        y_x_train = np.array(y_x_train)
        y_z_train = np.array(y_z_train)
        
        self.x_classifier = self._train_logistic_regression(X_x_train, y_x_train)
        self.z_classifier = self._train_logistic_regression(X_z_train, y_z_train)
        
        self.trained = True
    
    def _train_logistic_regression(self, X, y, n_iter=1000, lr=0.1):
        n_samples, n_features = X.shape
        n_outputs = y.shape[1]
        
        W = np.zeros((n_features, n_outputs))
        b = np.zeros(n_outputs)
        
        for _ in range(n_iter):
            logits = X @ W + b
            pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -10, 10)))
            
            error = pred - y
            grad_W = X.T @ error / n_samples
            grad_b = np.mean(error, axis=0)
            
            W -= lr * grad_W
            b -= lr * grad_b
        
        return {'W': W, 'b': b}
    
    def decode(self, x_syndrome, z_syndrome):
        if not self.trained:
            raise ValueError("Decoder not trained! Call train() first.")
        
        x_logits = x_syndrome @ self.x_classifier['W'] + self.x_classifier['b']
        x_pred = 1.0 / (1.0 + np.exp(-np.clip(x_logits, -10, 10)))
        x_correction = x_pred > 0.5
        
        z_logits = z_syndrome @ self.z_classifier['W'] + self.z_classifier['b']
        z_pred = 1.0 / (1.0 + np.exp(-np.clip(z_logits, -10, 10)))
        z_correction = z_pred > 0.5
        
        return x_correction, z_correction
    
    def apply_correction(self, surface_code, x_correction, z_correction):
        for i in range(self.n_data):
            if x_correction[i]:
                surface_code.apply_bit_flip(i)
            if z_correction[i]:
                surface_code.apply_phase_flip(i)
