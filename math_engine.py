import numpy as np

"""
Member 2 Module: Core Linear Algebra & Vector Space Operations.
Implemented from scratch using fundamental array arithmetic (without scikit-learn/scipy)
to demonstrate the mathematical workflow for UE25MA242A.
"""

def compute_dot_product(vec_a, vec_b):
    """
    Calculates the Dot Product (Inner Product) of two n-dimensional vectors:
    A . B = sum(A_i * B_i for i = 1..n)
    """
    return float(np.sum(vec_a * vec_b))

def compute_vector_norm(vec):
    """
    Calculates the L2 Norm (Euclidean Magnitude) of a vector:
    ||A|| = sqrt(A . A) = sqrt(sum(A_i^2))
    """
    dot_self = compute_dot_product(vec, vec)
    return float(np.sqrt(dot_self))

def compute_cosine_similarity(vec_a, vec_b):
    """
    Calculates Cosine Similarity between two vectors:
    cos(theta) = (A . B) / (||A|| * ||B||)
    Returns tuple: (cos_sim, dot_prod, norm_a, norm_b)
    """
    dot_prod = compute_dot_product(vec_a, vec_b)
    norm_a = compute_vector_norm(vec_a)
    norm_b = compute_vector_norm(vec_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0, dot_prod, norm_a, norm_b
        
    cos_sim = dot_prod / (norm_a * norm_b)
    # Clamp between -1.0 and 1.0 to avoid floating point precision issues in arccos
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return cos_sim, dot_prod, norm_a, norm_b

def compute_angle_degrees(cos_sim):
    """
    Calculates the geometric angle theta (in degrees) between two vectors:
    theta = arccos(cos_sim) * (180 / pi)
    """
    radians = np.arccos(np.clip(cos_sim, -1.0, 1.0))
    return float(np.degrees(radians))

def compute_euclidean_distance(vec_a, vec_b):
    """
    Calculates the Euclidean Distance ||A - B|| for comparison during Viva.
    """
    diff = vec_a - vec_b
    return compute_vector_norm(diff)
