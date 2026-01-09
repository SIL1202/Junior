# Sequence Recognition: LSTM vs. GRU Comparison

## 1. Project Overview & Architecture
The goal of this project is to classify sequences of numbers into five categories (A to E). These sequences represent patterns that the model needs to learn. In this experiment, I compared two different models, **LSTM** and **GRU**, to see which one is faster and more accurate for this specific task.

### Core Components
*   **Embedding Layer**: Since the input is just a list of IDs, I used an Embedding layer to turn these numbers into vectors. This helps the model understand the relationship between different tokens.
*   **Bidirectional Core:  I used **Bidirectional GRU** as my main model. "Bidirectional" means the model looks at the sequence from both the beginning to the end and the end to the beginning. This gives the model a full "view" of the data.
*   **Final Classifier**: I combined the results from both directions and used a Linear layer to pick the final category.
*   **Preprocessing**: I limited the sequence length to 128 (MAX_LEN) and used "Padding" to make all sequences the same size so the GPU can process them quickly.

## 2. Design Novelty: Comparative Study
- **Comparing Two Models**: Instead of just picking one model, I built and trained both **LSTM** and **GRU**. This allowed me to choose the best one based on real data, not just guessing.
- **AdamW Optimizer**: I used the AdamW optimizer instead of the standard Adam. AdamW is better at preventing the model from "overfitting" (meaning it learns the training data too well but fails on new data).
- **Smart Data Loading**: I created a custom way to load data that only pads the sequences when necessary, which saved a lot of computer memory.

## 3. Experimental Results
I trained both models for 10 Epochs. The results clearly demonstrate the advantages of the GRU architecture.

### Quantitative Comparison
| Metric | Bidirectional LSTM | Bidirectional GRU |
| :--- | :--- | :--- |
| **Final Training Loss** | 0.0800 | **0.0045** |
| **Final Validation Acc**| 98.62% | **99.75%** |
| **Convergence Speed** | Slower (fluctuated around Epoch 8) | **Fast** (Reached >99% by Epoch 2) |

### Visual Analysis
Looking at the **`loss_comparison.png`** and **`accuracy_comparison.png`** charts, the GRU model reached 99% accuracy in only 2 epochs. The **`cm_GRU.png` (Confusion Matrix)** also shows that the model made almost zero mistakes on all categories.

## 4. Difficulties Encountered

- **Index Error (KeyError: 2210)**: When I split the data into training and validation sets, the ID numbers got mixed up. This caused the code to crash because it couldn't find the right index. I fixed this by converting the data into a simple list (`tolist()`) so the index starts from 0 again.
- **JSON Data**: The data was in JSON format, which is different from standard CSV files. I had to use `pd.read_json` and find the exact file path on Kaggle to load it correctly.

## 5. Lessons Learned

- **Simpler is often Better**: I learned that even though LSTM is more complex, the GRU model was faster and more stable for this specific data.
- **Data Quality Matters**: I realized that how you prepare the data (like Padding and Indexing) is just as important as the model itself. Small mistakes in data handling can stop the whole model from working.