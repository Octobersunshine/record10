from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List, Optional, Dict, Any

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

app = FastAPI(title="DBSCAN/HDBSCAN Clustering API", description="基于DBSCAN和HDBSCAN的聚类API服务")


class ClusteringRequest(BaseModel):
    data_points: List[List[float]]
    eps: float = 0.5
    min_samples: int = 5


class ClusteringResponse(BaseModel):
    labels: List[int]
    n_clusters: int
    n_noise: int
    cluster_info: Optional[dict] = None
    border_points_reassigned: Optional[int] = None


class HDBSCANRequest(BaseModel):
    data_points: List[List[float]]
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "eom"


class ClusterTreeNode(BaseModel):
    id: int
    parent: Optional[int]
    children: List[int]
    lambda_val: float
    size: int
    is_cluster: bool


class HDBSCANResponse(BaseModel):
    labels: List[int]
    n_clusters: int
    n_noise: int
    probabilities: List[float]
    cluster_info: Optional[dict] = None
    cluster_hierarchy: Optional[List[ClusterTreeNode]] = None
    estimated_eps: Optional[float] = None
    min_spanning_tree: Optional[List[List[float]]] = None


def dbscan_with_stable_border_assignment(X, eps, min_samples):
    """
    改进的DBSCAN实现：边界点分配给距离最近的核心点所在的簇，
    消除数据顺序依赖。
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    core_indices = set(dbscan.core_sample_indices_)
    core_points = X[list(core_indices)]
    core_labels = labels[list(core_indices)]
    
    if len(core_points) == 0:
        return labels, 0
    
    border_points_reassigned = 0
    nbrs = NearestNeighbors(radius=eps)
    nbrs.fit(core_points)
    
    for i in range(len(X)):
        if i in core_indices:
            continue
            
        if labels[i] == -1:
            continue
            
        point = X[i].reshape(1, -1)
        distances, indices = nbrs.radius_neighbors(point)
        
        if len(indices[0]) > 0:
            min_dist_idx = np.argmin(distances[0])
            nearest_core_idx = indices[0][min_dist_idx]
            new_label = core_labels[nearest_core_idx]
            
            if labels[i] != new_label:
                labels[i] = new_label
                border_points_reassigned += 1
    
    return labels, border_points_reassigned


def estimate_optimal_eps(X, min_samples=5):
    """
    使用k-distance方法估计最优eps参数
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    sorted_distances = np.sort(distances[:, -1])
    optimal_eps = np.percentile(sorted_distances, 90)
    return float(optimal_eps)


def extract_cluster_hierarchy(hdbscan_model, n_samples):
    """
    从HDBSCAN模型中提取聚类层次树结构
    """
    if not hasattr(hdbscan_model, '_condensed_tree'):
        return None
    
    condensed_tree = hdbscan_model._condensed_tree
    cluster_labels = set(hdbscan_model.labels_)
    cluster_labels.discard(-1)
    
    tree_data = condensed_tree.to_numpy()
    
    cluster_node_map = {}
    node_id = 0
    
    for row in tree_data:
        parent = int(row[0])
        child = int(row[1])
        
        if child not in cluster_node_map:
            cluster_node_map[child] = node_id
            node_id += 1
        
        if parent not in cluster_node_map:
            cluster_node_map[parent] = node_id
            node_id += 1
    
    nodes_dict = {}
    
    for row in tree_data:
        parent = int(row[0])
        child = int(row[1])
        lambda_val = float(row[2])
        size = int(row[3])
        
        is_cluster = child in cluster_labels
        
        child_node_id = cluster_node_map[child]
        
        if child_node_id not in nodes_dict:
            nodes_dict[child_node_id] = ClusterTreeNode(
                id=child_node_id,
                parent=cluster_node_map.get(parent),
                children=[],
                lambda_val=lambda_val,
                size=size,
                is_cluster=is_cluster
            )
        else:
            nodes_dict[child_node_id].parent = cluster_node_map.get(parent)
            nodes_dict[child_node_id].lambda_val = lambda_val
            nodes_dict[child_node_id].size = size
            nodes_dict[child_node_id].is_cluster = is_cluster
    
    for row in tree_data:
        parent = int(row[0])
        child = int(row[1])
        
        parent_node_id = cluster_node_map[parent]
        child_node_id = cluster_node_map[child]
        
        if parent_node_id in nodes_dict and child_node_id not in nodes_dict[parent_node_id].children:
            nodes_dict[parent_node_id].children.append(child_node_id)
    
    nodes = list(nodes_dict.values())
    nodes.sort(key=lambda x: x.id)
    
    return nodes


@app.post("/hdbscan", response_model=HDBSCANResponse)
def perform_hdbscan(request: HDBSCANRequest):
    if not HDBSCAN_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="HDBSCAN未安装，请运行: pip install hdbscan"
        )
    
    if len(request.data_points) == 0:
        raise HTTPException(status_code=400, detail="数据点不能为空")
    
    try:
        X = np.array(request.data_points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="数据点必须是二维数组")
    
    try:
        min_samples = request.min_samples if request.min_samples is not None else request.min_cluster_size
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=request.min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=request.cluster_selection_epsilon,
            cluster_selection_method=request.cluster_selection_method,
            gen_min_span_tree=True
        )
        
        labels = clusterer.fit_predict(X)
        probabilities = clusterer.probabilities_.tolist()
        
        labels_list = labels.tolist()
        n_clusters = len(set(labels_list)) - (1 if -1 in labels_list else 0)
        n_noise = list(labels_list).count(-1)
        
        cluster_info = {}
        for label in set(labels_list):
            if label == -1:
                continue
            cluster_points = X[labels == label]
            cluster_info[f"cluster_{label}"] = {
                "size": len(cluster_points),
                "center": cluster_points.mean(axis=0).tolist(),
                "avg_probability": float(np.mean(clusterer.probabilities_[labels == label]))
            }
        
        estimated_eps = estimate_optimal_eps(X, min_samples)
        
        cluster_hierarchy = extract_cluster_hierarchy(clusterer, len(X))
        
        min_spanning_tree = None
        if hasattr(clusterer, 'minimum_spanning_tree_'):
            mst = clusterer.minimum_spanning_tree_
            min_spanning_tree = mst.tolist()
        
        return HDBSCANResponse(
            labels=labels_list,
            n_clusters=n_clusters,
            n_noise=n_noise,
            probabilities=probabilities,
            cluster_info=cluster_info,
            cluster_hierarchy=cluster_hierarchy,
            estimated_eps=estimated_eps,
            min_spanning_tree=min_spanning_tree
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HDBSCAN聚类过程出错: {str(e)}")


@app.post("/cluster", response_model=ClusteringResponse)
def perform_clustering(request: ClusteringRequest):
    if len(request.data_points) == 0:
        raise HTTPException(status_code=400, detail="数据点不能为空")
    
    try:
        X = np.array(request.data_points)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"数据格式错误: {str(e)}")
    
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail="数据点必须是二维数组")
    
    try:
        labels, reassigned = dbscan_with_stable_border_assignment(
            X, request.eps, request.min_samples
        )
        
        labels_list = labels.tolist()
        n_clusters = len(set(labels_list)) - (1 if -1 in labels_list else 0)
        n_noise = list(labels_list).count(-1)
        
        cluster_info = {}
        for label in set(labels_list):
            if label == -1:
                continue
            cluster_points = X[labels == label]
            cluster_info[f"cluster_{label}"] = {
                "size": len(cluster_points),
                "center": cluster_points.mean(axis=0).tolist()
            }
        
        return ClusteringResponse(
            labels=labels_list,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_info=cluster_info,
            border_points_reassigned=reassigned
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类过程出错: {str(e)}")


@app.get("/")
def read_root():
    return {
        "message": "DBSCAN/HDBSCAN Clustering API",
        "endpoints": {
            "/cluster": "标准DBSCAN聚类（边界点优化版）",
            "/hdbscan": "HDBSCAN层次密度聚类（自动选择eps）"
        },
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
