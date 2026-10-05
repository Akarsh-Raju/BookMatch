# Book-Match: Quantifying Novel Similarity via Feature Vectors

**Course:** UE25MA242A – Mathematical Foundation for AI & Data Science (MFAD 2026)  
**Project Level:** Experiential Learning Level 2 (Orange Problem)

---

## Problem Statement
Standard book recommendations often feel random and miss the specific blend of genres and tones a reader actually wants. This project maps 300 novels into 20-dimensional numerical feature vectors based on their genres and literary attributes. Using fundamental linear algebra operations—specifically dot products, vector norms, matrix rank, and cosine similarity—the application mathematically quantifies literary overlap and ranks the most similar novels alongside a 2D geometric vector projection dashboard.

---

## Mathematical Concepts Used
Each novel is represented as a vector $\vec{v} \in \mathbb{R}^{20}$ (12 genre dimensions + 8 literary attribute dimensions scaled between `0.0` and `1.0`).

1. **Feature Matrix & Rank ($\text{Rank}(X)$):**  
   The dataset forms a matrix $X \in \mathbb{R}^{300 \times 20}$. Computing $\text{Rank}(X) = 20$ confirms full column rank, proving that all 20 literary dimensions are linearly independent and non-redundant.
2. **Dot Product (Inner Product):**  
   Measures unnormalized feature overlap between target book $\vec{A}$ and candidate book $\vec{B}$:
   $$\vec{A} \cdot \vec{B} = \sum_{i=1}^{20} A_i B_i$$
3. **Euclidean ($L_2$) Norm:**  
   Measures the magnitude (intensity) of a novel's feature vector:
   $$\|\vec{A}\| = \sqrt{\vec{A} \cdot \vec{A}} = \sqrt{\sum_{i=1}^{20} A_i^2}$$
4. **Cosine Similarity & Geometric Angle ($\theta$):**  
   Normalizes the dot product by the product of magnitudes to measure the directional alignment between two novels, independent of vector length:
   $$\cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}, \quad \theta = \arccos(\cos(\theta)) \cdot \frac{180^\circ}{\pi}$$
5. **Euclidean Distance ($\|\vec{A} - \vec{B}\|$):**  
   Computed alongside Cosine Similarity in Stage 2 for comparative analysis during evaluation.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `novels_vector_dataset_300.csv` | Curated dataset of 300 novels across 12 genres and 8 literary attributes ($\mathbb{R}^{300 \times 20}$). |
| `data_loader.py` | Loads the CSV dataset using `pandas`, constructs the `numpy` feature matrix, and handles title lookup. |
| `math_engine.py` | Implements Dot Product, $L_2$ Norm, Cosine Similarity, Angle $\theta$, and Euclidean Distance from scratch. |
| `visualizer.py` | Projects high-dimensional vectors onto a 2D geometric plane ($\|\vec{B}\|\cos\theta, \|\vec{B}\|\sin\theta$) and renders the Top 5 dashboard. |
| `main.py` | Runs the interactive CLI pipeline and prints Stage 1, Stage 2, Stage 3, and Stage 4 outputs. |
| `requirements.txt` | Python dependencies (`numpy`, `pandas`, `matplotlib`). |

---

## How to Run Locally (VS Code / Terminal)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Sample Usage
When prompted:
```text
Enter a novel title: Six of Crows
```
The program will:
* **Stage 1:** Print the $300 \times 20$ feature matrix $X$, verify $\text{Rank}(X) = 20$, and display the target novel's vector $\vec{A}$ and norm $\|\vec{A}\|$.
* **Stage 2:** Display the intermediate linear algebra calculation table ($\vec{A} \cdot \vec{B}$, $\|\vec{B}\|$, $\cos(\theta)$, Angle $\theta^\circ$, and $\|\vec{A}-\vec{B}\|$).
* **Stage 3:** Output the final ranked list of recommended novels with percentage match scores.
* **Stage 4:** Open the interactive Matplotlib Geometric Vector Space & Similarity Dashboard window.
