# Music Scale Recognition Using Convolutional Neural Networks

1. ### Detailed Model Description
   - Core architecture: The model in this experiment is based on the ResNet-18 architecture. The reason for choosing it is because its ability to train deep networks efficiently through skip connections, which mitigate the vanishing gradient problem. 

   - Transfer learning: I utilized the pre-trained weights on the ImageNet dataset. This allows the model to recognize textures and patterns from the beginning.

   - Output Adaptation: Since the target in this homework is to recognize the 88 piano notes, I replaced the 1000-class Fully Connected Layer with a custom Linear layer featuring 88 output nodes.

   - Preprocessing: All of the input music Spectrograms were resized to $224 \times 224$ pixels and normalized.

2. ### Design Novelty
   In this homework, I chose ResNet-18 as the core architecture. Although it was originally designed for natural image recognition (e.g., cats and dogs), a music spectrogram is also an image rich in lines and textures, so the model can still be effectively applied. However, the residual connections alone could not completely eliminate the vanishing gradient problem.
   
    Because the piano has 88 keys, I replaced the original 1000-class output layer with a new 88-class fully connected layer. During preprocessing, I resized all spectrogram images to (224 \times 224) pixels and normalized the inputs to stabilize training.
    
    An interesting part of this homework is that we cannot use common data augmentation strategies for natural images. For example, we cannot flip the images. In a music spectrogram, the y-axis represents frequency, so flipping vertically would turn high tones into low tones, destroying the physical meaning of the data. Therefore, I only applied resizing and normalization, ensuring that the model learns the actual physical frequency patterns.

![Screenshot 2026-01-09 at 8.39.41 PM](https://hackmd.io/_uploads/Syb8P_0Ebg.png)
 The screenshot without flip the image
    
![Screenshot 2026-01-09 at 8.40.06 PM](https://hackmd.io/_uploads/r1LFwO0Ebx.png)
 The screenshot with flip the image
     
3. ### Achieved Results
   - Training Performance: After 10 epochs of training, the Cross-Entropy Loss showed a consistent downward trend, dropping significantly from the first epoch.
![Screenshot 2026-01-09 at 6.20.00 PM](https://hackmd.io/_uploads/rJAg5dAV-x.png)
   - Validation: The model demonstrated high accuracy in recognizing specific notes, even with similar-looking spectrogram patterns.
   - Kaggle Submission: The final output was saved to CNN_submission.csv according to the required sample format, ensuring correct label alignment.


4. ### Difficulties Encountered
   Although the dataset appears complete, I noticed that certain ranges still have relatively poor recognition performance. This may be due to weak feature patterns in the spectrogram or insufficient diversity of samples in those regions. To address this, I adjusted the batch size and monitored the validation loss curve. I found that overfitting started to occur after around the 15th epoch, so I decided to stop training at the 10th epoch to maintain better generalization.
   
5. ### Lessons Learned
   - Adaptability of Transfer Learning: I realized that pre-trained models like ResNet are surprisingly robust. Even though the original weights were trained on natural objects (cats, dogs), the early convolutional layers excel at detecting edges and textures. This proved perfectly applicable to the linear harmonic patterns found in music spectrograms, allowing for rapid convergence even with a domain shift.
   - Data Consistency: This project reinforced the importance of the order of samples in the testing dataset. Modifying the order would have resulted in a total mismatch with the ground truth, highlighting that data integrity is as important as model architecture.
   - Kaggle Workflow: I gained practical experience in using cloud-based notebooks and managing large datasets within the Kaggle platform.
