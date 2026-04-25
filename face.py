'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # Convert to numpy and uint8
    img_np = img.permute(1, 2, 0).numpy().astype('uint8')

    # Detect locations with the API
    face_locations = face_recognition.face_locations(img_np)

    # Iterate through the resulting tuple and convert from (top, right, bottom, left) to [x, y, width, height]
    for (top, right, bottom, left) in face_locations:
        x = float(left)
        y = float(top)
        width = float(right - left)
        height = float(bottom - top)
        
        detection_results.append([x, y, width, height])
    
    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    # STEP 1:
    # Extract feature vectors for imgs

    # Define the names list and the encodings list
    img_names = []
    encodings = []
    
    for img_name, img in imgs.items():
        # convert
        img_np = img.permute(1, 2, 0).numpy().astype('uint8')
        
        # detect face locations
        boxes = face_recognition.face_locations(img_np)
        
        # get 128 dimensions encodings
        face_encodings = face_recognition.face_encodings(img_np, boxes)
        
        # Guarding statement, in case no face was detected
        if len(face_encodings) > 0:
            encodings.append(torch.tensor(face_encodings[0], dtype=torch.float32))
        else:
            encodings.append(torch.zeros(128, dtype=torch.float32))
        
        img_names.append(img_name)
    
    # STEP 2:
    # Implement the K-means clustering myself

    # Build the encoding matrix of shape (n, 128)
    M = torch.stack(encodings)
    n = M.shape[0]

    # ABANDONNED: used kmeans++ to improve on the empty clusters problem
    # randomly pick K centroids as initialization
    #rand_indices = torch.randperm(n)[:K]
    #centroids = M[rand_indices].clone() # controids, shape (K, 128)


    centroids = kmeans_pp_init(M, K)

    N_ITERATIONS = 100
    cluster_assignments = torch.zeros(n, dtype=torch.long)

    for _ in range(N_ITERATIONS):
        # Compute distance matrix of shape (n, K)
        distances = torch.cdist(M, centroids)
        
        # assign the closest centroid: (n, K) -> (n,)
        cluster_assignments = torch.argmin(distances, dim=1)
        
        new_centroids = torch.zeros_like(centroids)
        
        # Update the center of each cluster to the average position
        for k in range(K):
            mask = (cluster_assignments == k)
            if mask.any():
                new_centroids[k] = M[mask].mean(dim=0)
            else:
                # Deal with empty centroids: assign the most far away one
                # the closest distancecs of points (N,)
                min_dists = distances.min(dim=1).values 
                farthest_idx = torch.argmax(min_dists).item()
                new_centroids[k] = M[farthest_idx].clone()
                
        
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        
        centroids = new_centroids

    for i, img_name in enumerate(img_names):
        cluster_idx = cluster_assignments[i].item()
        cluster_results[cluster_idx].append(img_name)

    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def kmeans_pp_init(data, K):
    # Random first center
    centers = [data[torch.randint(0, data.shape[0], (1,)).item()]]
    
    for _ in range(K - 1):
        # compute the center
        center_tensor = torch.stack(centers)
        dists = torch.cdist(data, center_tensor).min(dim=1).values

        # Squared dist
        dists_sq = dists ** 2
        probs = dists_sq / dists_sq.sum()

        # The further away, the more possible to be picked as center
        next_idx = torch.multinomial(probs, 1).item()
        centers.append(data[next_idx])
    
    return torch.stack(centers)
