import numpy as np

def upper_Triangle(A):

    A = np.array(A, dtype=float)
    n = A.shape[0]

    for k in range(n - 1):
        
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if A[pivot_row, k] == 0:
            raise ValueError("Matrix is singular; no unique solution.")
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]

        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:] # fast way to refactor for each column on the ith row , using the pivot_row

    return A 


# --- Quick test ---
if __name__ == "__main__":
    A = [[ 2,  1, -1],
         [-3, -1,  2],
         [-2,  1,  2]]
    
    U = np.array(upper_Triangle(A), dtype= float)
    for row in U:
        print("  ".join(f"{v:8.4f}" for v in row))

    #print("Upper Triangle of A :", U)
