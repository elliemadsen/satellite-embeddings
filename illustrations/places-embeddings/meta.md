Location
Latitude
Longitude
Elevation
Humidity
Tree Coverage
Embedding Vector
Atacama Desert, Chile
−24.3
−69.1
0.92
0.05
0.01
[0.92, 0.05, 0.01]
Amazon Rainforest, Brazil
−3.1
−60.0
0.35
0.94
0.98
[0.35, 0.94, 0.98]
Alpine Meadow, Swiss Alps
46.6
8.3
0.88
0.62
0.42
[0.88, 0.62, 0.42]
Coastal Mangroves, Vietnam
9.6
105.9
0.08
0.91
0.76
[0.08, 0.91, 0.76]

---

## Cosine Similarity Analysis

### Brazil vs. Vietnam (High Similarity)

**Embeddings:**

- Brazil: **b** = [0.35, 0.94, 0.98]
- Vietnam: **v** = [0.08, 0.91, 0.76]

**Cosine Similarity Equation:**

$$\cos(\theta) = \frac{\mathbf{b} \cdot \mathbf{v}}{|\mathbf{b}| \cdot |\mathbf{v}|}$$

**Calculation:**

$$\mathbf{b} \cdot \mathbf{v} = (0.35)(0.08) + (0.94)(0.91) + (0.98)(0.76) = 1.6282$$

$$|\mathbf{b}| = \sqrt{(0.35)^2 + (0.94)^2 + (0.98)^2} = 1.4023$$

$$|\mathbf{v}| = \sqrt{(0.08)^2 + (0.91)^2 + (0.76)^2} = 1.1883$$

$$\cos(\theta) = \frac{1.6282}{1.4023 \times 1.1883} = \mathbf{0.9771}$$

**Result:** Very high similarity (0.98) indicates Brazil and Vietnam locations share similar environmental characteristics (high humidity, tree coverage).

---

### Brazil vs. Chile (Low Similarity)

**Embeddings:**

- Brazil: **b** = [0.35, 0.94, 0.98]
- Chile: **c** = [0.92, 0.05, 0.01]

**Calculation:**

$$\mathbf{b} \cdot \mathbf{c} = (0.35)(0.92) + (0.94)(0.05) + (0.98)(0.01) = 0.3788$$

$$|\mathbf{b}| = 1.4023$$

$$|\mathbf{c}| = \sqrt{(0.92)^2 + (0.05)^2 + (0.01)^2} = 0.9214$$

$$\cos(\theta) = \frac{0.3788}{1.4023 \times 0.9214} = \mathbf{0.2932}$$

**Result:** Low similarity (0.29) indicates Brazil and Chile have quite different environmental profiles. Brazil is humid and forested; Chile (Atacama) is dry with minimal vegetation.
