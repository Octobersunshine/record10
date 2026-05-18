from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)


def kmeans_with_empty_cluster_handling(X, k, max_iter=300, tol=1e-4, random_state=42):
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    if k > n_samples:
        raise ValueError('k cannot be greater than number of samples')

    if k == n_samples:
        return np.arange(n_samples), X.copy()

    centroids = X[rng.choice(n_samples, k, replace=False)]

    for _ in range(max_iter):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        cluster_sizes = np.array([np.sum(labels == i) for i in range(k)])
        largest_cluster_idx = np.argmax(cluster_sizes)

        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                largest_cluster_points = X[labels == largest_cluster_idx]
                random_idx = rng.randint(0, len(largest_cluster_points))
                new_centroids[i] = largest_cluster_points[random_idx]
            else:
                new_centroids[i] = np.mean(cluster_points, axis=0)

        if np.all(np.abs(centroids - new_centroids) < tol):
            break

        centroids = new_centroids

    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)

    return labels, centroids


def calculate_sse(X, labels, centroids):
    sse = 0.0
    k = len(centroids)
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            distances = np.sum((cluster_points - centroids[i])**2)
            sse += distances
    return sse


def find_elbow_point(k_values, sse_values):
    if len(k_values) < 3:
        return k_values[0] if k_values else 1
    
    sse_array = np.array(sse_values)
    
    derivatives = np.diff(sse_array)
    
    second_derivatives = np.diff(derivatives)
    
    elbow_idx = np.argmax(second_derivatives) + 2
    
    return k_values[elbow_idx - 1]


def recommend_k_with_elbow(X, max_k=None, random_state=42):
    n_samples = len(X)
    
    if max_k is None:
        max_k = min(10, n_samples - 1)
    
    max_k = min(max_k, n_samples - 1)
    
    if max_k < 2:
        return 2, [], []
    
    k_values = list(range(1, max_k + 1))
    sse_values = []
    
    for k in k_values:
        labels, centroids = kmeans_with_empty_cluster_handling(X, k, random_state=random_state)
        sse = calculate_sse(X, labels, centroids)
        sse_values.append(sse)
    
    recommended_k = find_elbow_point(k_values, sse_values)
    
    return recommended_k, k_values, sse_values


@app.route('/kmeans', methods=['POST'])
def kmeans_clustering():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'points' not in data:
            return jsonify({'error': 'Missing "points" parameter'}), 400

        if 'k' not in data:
            return jsonify({'error': 'Missing "k" parameter'}), 400

        points = data['points']
        k = data['k']

        if not isinstance(points, list) or len(points) == 0:
            return jsonify({'error': 'Points must be a non-empty list'}), 400

        if not isinstance(k, int) or k <= 0:
            return jsonify({'error': 'k must be a positive integer'}), 400

        if k > len(points):
            return jsonify({'error': 'k cannot be greater than the number of points'}), 400

        X = np.array(points)

        if X.ndim != 2:
            return jsonify({'error': 'Points must be a 2D array (list of lists)'}), 400

        labels, centers = kmeans_with_empty_cluster_handling(X, k)

        return jsonify({
            'labels': labels.tolist(),
            'centers': centers.tolist(),
            'k': k,
            'num_points': len(points),
            'dimensions': X.shape[1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/kmeans/recommend-k', methods=['POST'])
def recommend_k():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if 'points' not in data:
            return jsonify({'error': 'Missing "points" parameter'}), 400

        points = data['points']
        max_k = data.get('max_k')

        if not isinstance(points, list) or len(points) == 0:
            return jsonify({'error': 'Points must be a non-empty list'}), 400

        X = np.array(points)

        if X.ndim != 2:
            return jsonify({'error': 'Points must be a 2D array (list of lists)'}), 400

        if len(points) < 3:
            return jsonify({'error': 'At least 3 points required for elbow method'}), 400

        recommended_k, k_values, sse_values = recommend_k_with_elbow(X, max_k)

        return jsonify({
            'recommended_k': recommended_k,
            'k_values': k_values,
            'sse_values': sse_values,
            'num_points': len(points),
            'dimensions': X.shape[1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
