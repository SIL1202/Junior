# Final Project Report

### Title: Music Scale Recognition Using Convolutional Neural Networks

1. #### Detailed Model Description
   The proposed model utilizes the ResNet-18 architecture as its foundation. ResNet (Residual Network) was selected for its ability to train deep networks efficiently through skip connections, which mitigate the vanishing gradient problem.

   - Feature Extraction: I employed Transfer Learning by utilizing weights pre-trained on the ImageNet dataset. This allows the model to recognize complex textures and geometric patterns from the start.
   - Output Adaptation: The original 1000-class output layer was replaced with a custom Linear layer featuring 88 output nodes, specifically corresponding to the 88 piano notes in the dataset.
   - Preprocessing: Input images (spectrograms) are normalized and resized to $224 \times 224$ pixels to remain consistent with the model's architecture.

2. #### Design Novelty
   The primary novelty lies in treating audio recognition as a visual problem. By transforming raw audio signals into mel-spectrograms (image data), we can leverage powerful computer vision techniques. I specifically used a low learning rate ($1e-4$) with the Adam optimizer to ensure that the pre-trained weights were "fine-tuned" rather than overwritten, allowing the model to adapt specifically to the unique visual signatures of different musical frequencies.

3. #### Achieved Results
   - Training Performance: After 10 epochs of training, the Cross-Entropy Loss showed a consistent downward trend, dropping significantly from the first epoch.
   - Validation: The model demonstrated high accuracy in recognizing specific notes, even with similar-looking spectrogram patterns.
   - Kaggle Submission: The final output was saved to CNN_submission.csv according to the required sample format, ensuring correct label alignment.

4. #### Difficulties Encountered
   - Path Management: One significant challenge was the directory structure in the Kaggle environment, specifically the distinction between hyphens (-) and underscores (_) in folder names (e.g., music-train vs music_train), which initially caused FileNotFoundError.
   - Resource Constraints: Without enabling GPU acceleration, the training time was prohibitively slow. Switching the "Accelerator" to GPU T4 x2 was necessary to complete the 10 epochs within a reasonable timeframe.

5. #### Lessons Learned
   - Transfer Learning Efficiency: I learned that pre-trained models are exceptionally robust; even when the target domain (music spectrograms) differs from the source domain (natural objects), the underlying feature extraction remains highly effective.
   - Data Consistency: This project reinforced the importance of the order of samples in the testing dataset. Modifying the order would have resulted in a total mismatch with the ground truth, highlighting that data integrity is as important as model architecture.
   - Kaggle Workflow: I gained practical experience in using cloud-based notebooks and managing large datasets within the Kaggle platform.
