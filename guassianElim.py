import numpy as np

def gaussian_elimination(A, b):

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # --- Forward elimination ---
    for k in range(n - 1):
      
        # initializing the pivot_row as the row that at the column k 
        # has the biggest aboslutevalue for each k row.
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        if A[pivot_row, k] == 0: # if the best candidate is 0 
            raise ValueError("Matrix is singular; no unique solution.")
        if pivot_row != k: # changing the pivot_row with the k row ( where we are ) so that we can eliminate below k ( which now is the pivot_row)  
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        # Eliminate entries below the pivot
        #Knowing that for the elimination a ROWi = ROWi -factor*ROWj
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back substituting 
    x = np.zeros(n)
  
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        # cleaner way to write it : 
        #known_terms = np.dot(A[i, i+1:] , x[i+1:])
        #x[i]=( b[i] - known_terms)) / A[i, i]
    return x


# --- Quick test ---
if __name__ == "__main__":
    A = [[ 2,  1, -1],
         [-3, -1,  2],
         [-2,  1,  2]]
    b = [8, -11, -3]

    x = gaussian_elimination(A, b)
    print("Solution:", x)
    print("Check A@x =", np.array(A) @ x)  
