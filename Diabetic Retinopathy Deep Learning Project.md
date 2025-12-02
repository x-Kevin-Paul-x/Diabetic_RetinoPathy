

# **Advanced Deep Learning Methodologies for the Detection and Grading of Diabetic Retinopathy: A Comprehensive Technical Strategy Report**

## **1\. Clinical and Pathophysiological Foundations of Diabetic Retinopathy**

The successful development of artificial intelligence systems for medical diagnostics necessitates a profound understanding of the underlying pathology. In the domain of Diabetic Retinopathy (DR), the machine learning engineer must transition from viewing images as mere arrays of pixel intensities to understanding them as biological landscapes scarred by metabolic dysfunction. DR is not a singular event but a progressive microvascular complication of diabetes mellitus, characterized by a cascade of damage to the retinal blood vessels caused by chronic hyperglycemia. To build a robust predictive model, one must first comprehend the clinical features that the neural network is expected to extract and quantify.

### **1.1 The Microvascular Cascade and Clinical Lesions**

The retina is a highly metabolically active tissue, requiring a dense network of capillaries to supply oxygen and nutrients. Chronic high blood sugar levels initiate a deleterious chain of events, primarily damaging the pericytes—contractile cells that wrap around the endothelial cells of capillaries to regulate blood flow and vessel stability. The loss of pericytes weakens the vessel walls, leading to the formation of **microaneurysms**, which are the earliest clinically visible signs of DR.1 These appear on fundus photography as small, deep-red dots, often indistinguishable to the untrained eye from small hemorrhages or pigment spots. For a deep learning model, the detection of microaneurysms represents a "small object detection" problem, challenging the spatial resolution capabilities of architectures that aggressively downsample input images.

As the vessel walls continue to compromise, the blood-retina barrier breaks down, leading to leakage. When blood escapes into the retinal tissue, it manifests as **hemorrhages**. The morphology of these hemorrhages provides critical depth information: "dot-and-blot" hemorrhages occur in the deeper retinal layers where cells are vertically oriented, while "flame-shaped" hemorrhages occur in the superficial nerve fiber layer.1 Simultaneously, the leakage of plasma rich in lipoproteins results in the deposition of **hard exudates**. These appear as distinct, bright yellow or white waxy lesions with sharp margins, often arranged in circinate (circular) patterns surrounding leaking microaneurysms.1 For a Convolutional Neural Network (CNN), distinguishing hard exudates from the optic disc (which is also bright) or cotton wool spots is a classic texture discrimination task.

**Cotton wool spots**, unlike hard exudates, are soft, fluffy white lesions with indistinct margins. They do not represent leakage but rather localized infarction (tissue death) of the nerve fiber layer caused by occlusion of precapillary arterioles.1 Their presence indicates ischemia (lack of blood flow) and marks a transition toward more severe disease states.

In the most advanced stages, the retina, starved of oxygen, secretes Vascular Endothelial Growth Factor (VEGF) to stimulate the growth of new blood vessels—a process known as **neovascularization**. This defines **Proliferative Diabetic Retinopathy (PDR)**. These new vessels are fragile and prone to rupture, leading to vitreous hemorrhage and tractional retinal detachment, which are the primary causes of severe vision loss.2 The detection of neovascularization is notoriously difficult for automated systems as these vessels are fine, irregular, and often obscure, requiring the model to learn high-frequency spatial features distinct from the normal vasculature.

### **1.2 The International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale**

To standardize patient management, the global ophthalmology community adopted the International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale. This scale serves as the "ground truth" schema for the majority of machine learning datasets, including the benchmark APTOS 2019\.5 It classifies the disease into five ordinal stages, a crucial detail that dictates the choice of loss functions in predictive modeling. The classes are not categorical buckets but represent a continuum of severity.

**Table 1: The ICDR Disease Severity Scale and Feature Mapping**

| Grade | Severity Level | Clinical Definition & Machine Learning Features |
| :---- | :---- | :---- |
| **0** | No Apparent Retinopathy | **Clinical:** No abnormalities visible. **ML Challenge:** The model must learn the "normal" variance of retinal anatomy (optic disc shape, vessel tortuosity) to avoid false positives from artifacts or dust on the lens. |
| **1** | Mild Non-Proliferative DR (NPDR) | **Clinical:** Presence of microaneurysms only. **ML Challenge:** Requires high-resolution input to detect minute red dots (often \< 30 microns). Heavily penalized if misclassified as Normal (Grade 0). |
| **2** | Moderate NPDR | **Clinical:** More than just microaneurysms but less than Severe NPDR. Includes dot/blot hemorrhages, hard exudates, and cotton wool spots. **ML Challenge:** The "catch-all" category between mild and severe. High intra-class variance makes this arguably the hardest class to bound decision boundaries for. |
| **3** | Severe NPDR | **Clinical:** The "4-2-1 Rule" applies. Any of: 1\. Severe intraretinal hemorrhages in all 4 quadrants. 2\. Definite venous beading in 2+ quadrants. 3\. Intraretinal Microvascular Abnormalities (IRMA) in 1+ quadrant. **ML Challenge:** Requires *global context* and *counting*. The model cannot just detect a feature; it must count its occurrence across specific spatial quadrants of the eye. |
| **4** | Proliferative DR (PDR) | **Clinical:** Neovascularization, vitreous/preretinal hemorrhage. **ML Challenge:** Detecting fine new vessels or large obscuring hemorrhages. The distinction from Severe NPDR is critical for immediate surgical intervention. |

Source: Synthesized from.1

It is vital to emphasize the **4-2-1 rule** for Severe NPDR.2 This rule introduces a complexity that standard CNNs often struggle with: spatial reasoning. A CNN is translation invariant; it identifies a hemorrhage regardless of where it is. However, to distinguish Grade 2 from Grade 3, the location (quadrant) and the frequency (count \> 20\) matter.4 This suggests that architectures capable of global attention, such as Vision Transformers or attention-augmented CNNs, may have a theoretical advantage in strictly adhering to clinical grading criteria compared to pure CNNs.

### **1.3 Distinguishing Diabetic Macular Edema (DME)**

While Diabetic Retinopathy refers to the overall vascular status of the retina, Diabetic Macular Edema (DME) is a specific complication where fluid accumulates in the macula—the central part of the retina responsible for sharp vision. The ICDR scale includes a separate classification for DME, graded based on the proximity of hard exudates to the foveal center.1 While some datasets like IDRiD provide DME labels 7, the primary objective of most DR screening projects (and this report) is the 5-stage DR severity grading. However, the presence of hard exudates (a key feature of DME) correlates strongly with DR severity, meaning features learned for DME detection often reinforce DR grading performance.8

---

## **2\. The Data Ecosystem: Benchmarks, Bias, and Acquisition**

In the realm of medical deep learning, data is not merely a resource; it is the primary constraint and the defining factor of model performance. Unlike natural image tasks (e.g., ImageNet), where classes are distinct (cat vs. dog) and balanced, medical datasets are characterized by high inter-class similarity, extreme class imbalance, and significant variability in acquisition hardware. A successful project execution plan relies on a strategic combination of available datasets.

### **2.1 The APTOS 2019 Blindness Detection Dataset**

The APTOS 2019 dataset, provided by the Aravind Eye Hospital in India, stands as the current gold standard for benchmarking DR models.9 It comprises 3,662 training images and 1,928 test images.

* **Acquisition Diversity:** Unlike standardized clinical trials, these images were captured using various fundus cameras in rural screening camps. This introduces "domain shift" within the dataset itself—variations in lighting, color temperature, and field of view are significant.  
* **Class Distribution:** The dataset exhibits the typical "long-tail" distribution seen in medical screening. The majority of cases are "No DR" (Grade 0\) or "Moderate DR" (Grade 2), with significantly fewer examples of "Severe" (Grade 3\) and "Proliferative" (Grade 4\) disease.5 Specifically, Grade 0 has \~1,805 images, while Grade 3 has only \~193 images.11 This 10:1 imbalance necessitates aggressive sampling strategies or weighted loss functions to prevent the model from collapsing into a trivial predictor of the majority class.  
* **Label Quality:** The images were graded by a panel of ophthalmologists, but the subjective nature of the ICDR scale (e.g., distinguishing "moderate" from "severe" hemorrhages) introduces inherent label noise.

### **2.2 The EyePACS (Kaggle 2015\) Dataset**

The EyePACS dataset is massive, containing approximately 88,702 images (35k train, 53k test).13

* **Strategic Utility:** Due to its size, EyePACS is invaluable for **pre-training**. A common strategy is to train a model on EyePACS to learn robust retinal feature extractors and then fine-tune it on the smaller, curated APTOS dataset.14  
* **Noise Profile:** The dataset is notorious for its quality issues. It contains images that are out of focus, overexposed, or contain artifacts (e.g., dust on the lens). Furthermore, the class imbalance is even more extreme than in APTOS, with over 73% of images being Grade 0\.13 Training solely on EyePACS without filtering can lead to models that generalize poorly to high-quality clinical data.

### **2.3 High-Precision Datasets: Messidor-2 and IDRiD**

To validate a model's clinical robustness, one must test on data from a completely different distribution (e.g., different hospital system, different population).

* **Messidor-2:** This French dataset contains 1,748 images (874 pairs). It is renowned for its image quality and macula-centered framing.16 It serves as an excellent external validation set to test if a model trained on Indian data (APTOS) generalizes to a European population.  
* **IDRiD (Indian Diabetic Retinopathy Image Dataset):** While small (516 images), IDRiD is unique because it offers **pixel-level annotations** (segmentation masks) for lesions like microaneurysms and hemorrhages.7 This allows for the training of auxiliary segmentation tasks or "attention guidance," where the model is explicitly penalized for not looking at the pathology.

### **2.4 Data Augmentation Strategies**

Given the paucity of data in the severe classes, augmentation is mandatory. However, medical images must be augmented with care to preserve anatomical validity.

* **Geometric Transforms:** Retinal images are rotationally invariant (the disease severity does not change if the image is rotated). Therefore, random rotations (0-360 degrees), horizontal/vertical flips, and slight scaling are "safe" and highly effective augmentations.19  
* **Photometric Transforms:** Color jitter (brightness, contrast, saturation, hue) is critical to make the model robust to different camera sensors. However, excessive color shift can be dangerous; shifting the hue too far might make a red hemorrhage look like a dark pigment scar, altering the diagnosis.  
* **Elastic Deformations:** Libraries like MONAI provide elastic deformations that simulate biological tissue variations, which can be more effective than simple affine transforms for medical imaging.21

---

## **3\. Preprocessing: The "Ben Graham" Standard**

In deep learning for computer vision, preprocessing is often limited to resizing and normalization. In retinal imaging, however, specific domain-driven preprocessing techniques are required to normalize the high variability in lighting conditions caused by the spherical geometry of the eye and the aperture of the camera. The industry standard, widely adopted following the 2015 Kaggle competition, is the **Ben Graham Method**.22

### **3.1 The Problem of Vignetting and Luminosity**

Fundus images are typically captured through a circular aperture, resulting in a bright central region (macula/optic disc) and a significantly darker periphery. This "vignetting" effect obscures lesions located near the edges of the retina. Furthermore, the aspect ratio of the fundus may vary, with black borders occupying a significant portion of the image tensor, wasting computational resources.

### **3.2 The Ben Graham Algorithm**

The method aims to normalize luminosity and texture across the image, highlighting high-frequency features (lesions) while suppressing the slowly varying background (illumination gradients).

Step 1: Crop and Centering  
The first operation is to remove the uninformative black background. This is achieved by thresholding the image to find the mask of the eye and cropping the bounding box around it.

* *Implementation Note:* It is crucial to enforce a square aspect ratio after cropping to prevent distortion during resizing, as biological structures (like the circular optic disc) should not be warped into ovals.22

Step 2: Gaussian Blur Subtraction (Local Average Subtraction)  
This is the core of the technique. The image is blended with a Gaussian-blurred version of itself. The mathematical operation can be described as:

$$I\_{processed} \= \\alpha \\cdot I\_{original} \+ \\beta \\cdot G(I\_{original}, \\sigma) \+ \\gamma$$

Where $G(I, \\sigma)$ is the Gaussian blur of the image with kernel size $\\sigma$. Typically, $\\alpha=4$, $\\beta=-4$, and $\\gamma=128$ (for 0-255 pixel values) are used.22

* **Mechanism:** By subtracting the blurred version (which represents the low-frequency illumination component) from the original, the operation acts as a **high-pass filter**. This removes the lighting variation (vignetting) and enhances the contrast of fine details like microaneurysms and vessels.  
* **Visual Result:** The resulting image often appears strictly grey/orange with very high contrast, where lesions pop out significantly against the background.

Step 3: Circular Masking  
Finally, a circular mask is applied to the image to ensure that the boundaries are sharp and that the model does not learn artifacts from the camera's edge.24

### **3.3 Comparative Efficacy**

Studies comparing Ben Graham preprocessing against standard resizing or Contrast Limited Adaptive Histogram Equalization (CLAHE) have consistently shown that the Ben Graham method yields faster convergence and higher validation accuracy.24 While CLAHE enhances local contrast, it can sometimes over-amplify noise in the dark regions of the retina, leading to false positives. The Gaussian subtraction method is more robust to this noise amplification.

---

## **4\. Architectural Paradigms: From CNNs to Vision Transformers**

The choice of model architecture is pivotal. While early approaches utilized VGG or Inception networks, the landscape in 2024-2025 is dominated by EfficientNets and emerging Hybrid Vision Transformers.

### **4.1 The CNN Backbone: EfficientNet Supremacy**

For the majority of practical applications and competition leaderboards, **EfficientNet** remains the architecture of choice.

* **Mechanism:** Introduced by Google Research, EfficientNet optimizes performance through **Compound Scaling**. Unlike ResNet, which scales primarily by adding layers (depth), EfficientNet scales depth, width (number of channels), and resolution simultaneously using a set of fixed coefficients.26  
* **Relevance to DR:** The resolution scaling is particularly critical for Diabetic Retinopathy. Detecting a microaneurysm (often just a few pixels wide) requires high-resolution input. EfficientNet-B5, for instance, operates at a native resolution of 456x456, while EfficientNet-B7 goes up to 600x600.  
* **Performance:** Empirical benchmarks on the APTOS dataset consistently show EfficientNet-B5 and B6 outperforming ResNet-50 and ResNet-101 by significant margins (e.g., Kappa scores of 0.906 vs 0.801).27 The architecture's efficiency also allows for larger batch sizes, which helps stabilize the training of Batch Normalization layers—a crucial factor in medical imaging where batch sizes are often constrained by GPU memory.

### **4.2 The Challenge of Global Context: Enter Vision Transformers (ViT)**

While CNNs excel at detecting local features (a hemorrhage at pixel $x,y$), they struggle with global relationships. Recall the **4-2-1 rule** for Severe NPDR: "Hemorrhages in 4 quadrants." A CNN must aggregate local detections into a global count, which requires deep layers with very large effective receptive fields.

* **Vision Transformers (ViT):** ViTs process images as sequences of patches and use Self-Attention mechanisms to relate every patch to every other patch. This theoretically allows the model to instantly compare the superior-nasal quadrant with the inferior-temporal quadrant, making it ideal for the 4-2-1 rule.29  
* **The Data Constraint:** The limitation of pure ViTs is their lack of inductive bias (they don't "know" about local edges like CNNs do). Consequently, they require massive datasets to train effectively. Training a ViT from scratch on the small 3,662-image APTOS dataset often leads to overfitting and poor performance compared to CNNs.29

### **4.3 The Hybrid Solution: Convolutional Vision Transformers (CvT)**

The current State-of-the-Art (SOTA) research advocates for **Hybrid Architectures**. These models use a CNN stem (like the first few blocks of a ResNet or EfficientNet) to extract low-level features (edges, textures, vessels) and then pass these feature maps to a Transformer backend to handle the global reasoning (classification based on spatial distribution).

* **Advantages:** This approach combines the data efficiency of CNNs (due to translation invariance) with the global reasoning capability of Transformers. Models like **CvT (Convolutional Vision Transformer)** have demonstrated higher Kappa scores (0.84+) on DR datasets compared to standalone CNNs or ViTs.29

### **4.4 Pooling Strategies: Beyond Max and Average**

In standard classification, the final feature map is collapsed into a vector using Global Average Pooling (GAP). For DR, this is suboptimal. Averaging the features of a small, distinct lesion (like a neovascular frond) with the vast healthy retina surrounding it can "wash out" the signal.

* Generalized Mean (GeM) Pooling: Top-performing solutions (such as the 1st place APTOS solution) replace GAP with GeM Pooling. GeM is a trainable pooling layer that can learn to focus on high-activation areas (like Max Pooling) or spread attention (like Average Pooling) depending on the task.

  $$f \= \\left( \\frac{1}{|\\Omega|} \\sum\_{x \\in \\Omega} x^p \\right)^{1/p}$$

  Where $p$ is a learnable parameter. As $p \\to \\infty$, it behaves like Max Pooling; as $p \\to 1$, it behaves like Average Pooling.31 This adaptability allows the model to preserve the signal of small, critical lesions.

---

## **5\. The Mathematics of Optimization: Loss Functions and Metrics**

Perhaps the most critical technical decision in a DR project is the formulation of the learning objective. The metric used for evaluation is **Quadratic Weighted Kappa (QWK)**, but the loss function used for training must be carefully selected to optimize this non-differentiable metric.

### **5.1 The Metric: Quadratic Weighted Kappa (QWK)**

Accuracy is a dangerous metric in ordinal classification. If the ground truth is Grade 4 (PDR), a prediction of Grade 0 (No DR) is a disastrous medical error, whereas a prediction of Grade 3 (Severe) is a minor staging error. Standard accuracy penalizes both equally (0/1 loss).  
QWK solves this by applying a quadratic penalty weight matrix $W$:

$$W\_{i,j} \= \\frac{(i-j)^2}{(N-1)^2}$$

where $i$ is the true class and $j$ is the predicted class. The score is calculated as $\\kappa \= 1 \- \\frac{\\sum W\_{ij} O\_{ij}}{\\sum W\_{ij} E\_{ij}}$, where $O$ is the observed confusion matrix and $E$ is the expected matrix by chance.32

* **Implication:** The model must be punished heavily for "far" misses.

### **5.2 Loss Function Strategies**

Since QWK is not differentiable, we cannot use it directly in backpropagation (though differentiable approximations exist, they are often unstable 34).

Strategy A: Regression with Threshold Optimization (The SOTA Approach)  
Instead of a 5-class Softmax output, the model is modified to output a single scalar value (linear activation). The problem is treated as regression.

1. **Loss:** Use **Mean Squared Error (MSE)** or **Smooth L1 Loss**. These losses naturally enforce the ordinal relationship: predicting 0 when truth is 4 results in a huge loss ($4^2=16$), whereas predicting 3 results in a small loss ($1^2=1$).  
2. **Inference:** The scalar output (e.g., 2.3) must be converted to a class. While simple rounding (0.5, 1.5, 2.5) works, it assumes equidistant classes.  
3. **Threshold Optimization:** A post-processing step using an optimizer (like scipy.optimize.minimize) finds the optimal thresholds $\[t\_0, t\_1, t\_2, t\_3\]$ that maximize QWK on the validation set. This technique was key to the 1st place solution in the APTOS competition.27

Strategy B: Ordinal Regression (Rank Consistency)  
This method transforms the classification into a series of binary sub-problems: "Is the grade \> 0?", "Is the grade \> 1?", "Is the grade \> 2?", etc.

* **Implementation:** Libraries like spacecutter or custom "Coral" layers implement this. The model outputs $K-1$ logits.  
* **Benefit:** This avoids the assumption that the "distance" between Grade 0 and 1 is the same as between Grade 3 and 4 (which regression assumes), while still enforcing the order.35

Strategy C: Focal Loss / Weighted Cross-Entropy  
If treating it as classification, one must use Weighted Cross-Entropy or Focal Loss to handle the extreme class imbalance (10:1 ratio between Grade 0 and Grade 4). Focal Loss dynamically scales the loss based on the confidence of the prediction, forcing the model to focus on "hard" examples (minority classes) rather than the easy Grade 0 cases.19  
**Recommendation:** The **Regression \+ Optimized Thresholds** strategy generally yields the highest QWK scores on leaderboards and is recommended for the primary model, potentially ensembled with a classification model trained with Focal Loss.

---

## **6\. Explainability (XAI): Bridging the Gap to Clinical Adoption**

A Deep Learning model, no matter how accurate, is a "black box." In ophthalmology, a clinician cannot accept a diagnosis of "Proliferative DR" without visual evidence. Explainable AI (XAI) is therefore not an optional feature but a mandatory requirement for deployment.

### **6.1 Saliency Mapping: Grad-CAM**

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is the standard technique for visualizing CNN decisions. It uses the gradients of the target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image.39

* **Validation:** For a correct prediction of DR, the Grad-CAM heatmap should overlap with the clinical lesions (hemorrhages, exudates).  
* **Failure Analysis:** If the heatmap highlights the optic disc, the eyelids, or the black border, the model is likely learning spurious correlations ("Clever Hans" effect). For example, a model might learn that images from a specific hospital (identified by a specific border artifact) always have severe disease.40

### **6.2 Pixel Attribution: Integrated Gradients**

While Grad-CAM provides a "blob" of attention, **Integrated Gradients (IG)** provides pixel-level attribution. It calculates the integral of gradients with respect to inputs along the path from a baseline (black image) to the input image.

* **Utility:** IG is particularly useful for identifying fine features like neovascularization that might be lost in the coarse upsampling of Grad-CAM.42 It satisfies the axiom of sensitivity (if an input change changes the output, the attribution is non-zero), making it theoretically more robust than simple saliency maps.

---

## **7\. Professional Project Structure and Tooling**

To execute a project of this complexity, ad-hoc scripts are insufficient. A modular, reproducible, and scalable software engineering approach is required.

### **7.1 Tooling Stack**

* **Framework:** **PyTorch** is strongly recommended over TensorFlow for this specific domain. The majority of SOTA implementations (EfficientNet, ViT, Hybrid) and medical libraries (MONAI) are PyTorch-first. Its dynamic graph nature facilitates the debugging of complex custom loops like Ordinal Regression.43  
* **Configuration:** **Hydra**. Unlike argparse, Hydra allows for hierarchical configuration using YAML files. This is essential for managing the combinatorial explosion of hyperparameters (backbone choice, image size, learning rate, loss function type) without modifying code.45  
* **Experiment Tracking:** **Weights & Biases (W\&B)**. W\&B offers superior visualization capabilities for image data (logging validation predictions alongside Grad-CAM heatmaps) compared to MLflow. Its "sweeps" feature is also excellent for hyperparameter optimization.47  
* **Medical Library:** **MONAI (Medical Open Network for AI)**. Use MONAI for its robust, medically-validated data loaders and augmentations. It handles nuances of medical data formats and provides deterministic transformations essential for reproducibility.21

### **7.2 Directory Structure**

The following structure promotes separation of concerns and reproducibility:

diabetic\_retinopathy\_project/  
├── conf/ \# Hydra Configuration  
│ ├── config.yaml \# Main entry point  
│ ├── dataset/ \# Dataset specific configs  
│ │ ├── aptos.yaml  
│ │ └── eyepacs.yaml  
│ ├── model/ \# Model architectures  
│ │ ├── efficientnet\_b5.yaml  
│ │ └── cvt\_w24.yaml  
│ ├── loss/ \# Loss function configs  
│ │ └── mse\_smooth.yaml  
│ └── training/ \# Training hyperparameters  
│ └── default.yaml  
├── data/  
│ ├── raw/ \# Immutable raw data  
│ ├── processed/ \# Ben Graham processed images  
│ └── splits/ \# K-Fold CSV files (seed fixed)  
├── notebooks/ \# EDA and prototyping  
│ ├── 01\_distribution\_analysis.ipynb  
│ └── 02\_gradcam\_visualization.ipynb  
├── src/ \# Core logic  
│ ├── init.py  
│ ├── datamodules/ \# PyTorch/Lightning DataModules  
│ │ ├── aptos\_loader.py  
│ │ └── transforms.py \# Albumentations/MONAI logic  
│ ├── models/  
│ │ ├── backbones.py \# timm wrappers  
│ │ └── heads.py \# Regression/Ordinal heads  
│ ├── utils/  
│ │ ├── ben\_graham.py \# Preprocessing implementation  
│ │ └── metrics.py \# QWK calculation  
│ └── trainer.py \# LightningModule definition  
├── train.py \# Entry point for training  
├── inference.py \# Deployment inference script  
├── requirements.txt  
└── README.md

---

## **8\. Thorough Execution Plan**

This roadmap assumes a 10-week timeline for a single researcher or a small team, moving from data ingestion to a deployed prototype.

### **Phase 1: Foundations & Pipeline (Weeks 1-2)**

* **Data Ingestion:** Download APTOS 2019 and EyePACS datasets.  
* **Preprocessing Implementation:** Implement ben\_graham.py. Create a script to process all images offline to 512x512 resolution (saving as PNG to avoid compression artifacts).  
* **Split Generation:** Create a StratifiedKFold (k=5) split for APTOS. This is crucial: random splitting will lead to unrepresentative validation sets due to the class imbalance.10  
* **DataLoader:** Build a MONAI/PyTorch Dataset class. Implement on-the-fly augmentations: HorizontalFlip, VerticalFlip, RandomRotate90, and moderate ColorJitter.

### **Phase 2: Baseline & Metric Stabilization (Weeks 3-4)**

* **Baseline Model:** Train a ResNet18 using standard Cross-Entropy Loss.  
* **Metric Integration:** Implement the Quadratic Weighted Kappa (QWK) calculation in the validation loop.  
* **Tracking:** Initialize Weights & Biases. Log Training Loss, Validation Loss, Accuracy, and QWK per epoch.  
* **Sanity Check:** Ensure the baseline achieves a QWK \> 0.0. If QWK stays near 0 or is negative, check the label encoding or the confusion matrix (the model is likely predicting only the majority class).

### **Phase 3: The SOTA Pursuit (Weeks 5-7)**

* **Architecture Upgrade:** Switch to EfficientNet-B5 (pre-trained on ImageNet).  
* **Loss Function Pivot:** Switch from Cross-Entropy to **Smooth L1 Loss (Regression)**. Change the model output to a single neuron (linear activation).  
* **External Pre-training:** (Optional but recommended) Train the model first on the 88k EyePACS images for 10-15 epochs, then fine-tune on APTOS for 20-30 epochs. This "double transfer learning" typically boosts Kappa by 0.02-0.05.  
* **Threshold Optimization:** Implement the scipy.optimize routine to find the best cutoffs for the regression output at the end of each validation epoch.

### **Phase 4: Refinement & Explainability (Weeks 8-9)**

* **Ensembling:** Train 3-5 models (e.g., EfficientNet-B5, EfficientNet-B4, and a Hybrid ViT/CvT) using different random seeds or folds. Average their regression outputs before thresholding. Ensembling is the most reliable way to boost QWK.31  
* **XAI Pipeline:** Create a notebook to run the best model on the Validation set and generate Grad-CAM overlays. Manually inspect misclassified "Severe" cases. Are they subtle? Are they labeling errors?

### **Phase 5: Deployment Strategy (Week 10\)**

* **Model Export:** Export the trained PyTorch model to **ONNX** format. This allows for framework-agnostic deployment and optimization (e.g., using TensorRT or ONNX Runtime).49  
* **Quantization:** Apply dynamic quantization (Float32 \-\> Int8) to reduce model size (\~40MB for EfficientNet) and latency, enabling deployment on edge devices or standard CPUs.49  
* **API Construction:** Wrap the ONNX runtime session in a **FastAPI** service. FastAPI is preferred over TorchServe for single-model microservices due to its lower overhead and high performance (asynchronous handling).50  
* **Dockerization:** Build a Docker container containing the API code and the ONNX model file. This ensures the model can be deployed identically on a local server or a cloud instance (AWS/GCP).

## **9\. Conclusion**

The development of an automated grading system for Diabetic Retinopathy is a challenge that intersects complex pathology with advanced computer vision. Success does not lie in simply applying a "stock" CNN to raw images. It requires a nuanced approach: standardizing inputs via Ben Graham preprocessing to mitigate domain shift, leveraging EfficientNet architectures with compound scaling to resolve minute lesions, and, crucially, treating the problem as an ordinal regression task to align the loss function with the clinical reality of disease progression. By adhering to the structured methodologies and XAI integrations outlined in this report, one can build a system that is not only statistically performant but clinically transparent and trustworthy.

#### **Works cited**

1. Comparing the International Clinical Diabetic Retinopathy (ICDR) severity scale \- PMC \- NIH, accessed on November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10436766/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10436766/)  
2. table 1 diabetic retinopathy disease severity scale and international clinical diabetic retinopathy disease severity scale, accessed on November 30, 2025, [https://www.aao.org/asset.axd?id=b465da67-9d41-4061-8d65-328b54310f03](https://www.aao.org/asset.axd?id=b465da67-9d41-4061-8d65-328b54310f03)  
3. Classification of diabetic retinopathy and diabetic macular edema \- PMC \- PubMed Central, accessed on November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC3874488/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3874488/)  
4. International Clinical Diabetic Retinopathy Severity Scale | PDF | Vision \- Scribd, accessed on November 30, 2025, [https://www.scribd.com/document/111181351/International-Clinical-Diabetic-Retinopathy-Severity-Scale](https://www.scribd.com/document/111181351/International-Clinical-Diabetic-Retinopathy-Severity-Scale)  
5. Class distribution in the APTOS 2019 dataset. Image generated using VS code Abbreviation: DR: Diabetic retinopathy. \- ResearchGate, accessed on November 30, 2025, [https://www.researchgate.net/figure/Class-distribution-in-the-APTOS-2019-dataset-Image-generated-using-VS-code-Abbreviation\_fig2\_383831910](https://www.researchgate.net/figure/Class-distribution-in-the-APTOS-2019-dataset-Image-generated-using-VS-code-Abbreviation_fig2_383831910)  
6. Classification of diabetic retinopathy – GPnotebook, accessed on November 30, 2025, [https://gpnotebook.com/en-IE/pages/ophthalmology/classification-of-diabetic-retinopathy](https://gpnotebook.com/en-IE/pages/ophthalmology/classification-of-diabetic-retinopathy)  
7. IDRiD: Diabetic Retinopathy – Grading \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset](https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset)  
8. Diabetic retinopathy image classification method based on GreenBen data augmentation \- arXiv, accessed on November 30, 2025, [https://arxiv.org/pdf/2410.09444](https://arxiv.org/pdf/2410.09444)  
9. APTOS 2019 blindness detection competition dataset \- LDM, accessed on November 30, 2025, [https://service.tib.eu/ldmservice/dataset/aptos-2019-blindness-detection-competition-dataset](https://service.tib.eu/ldmservice/dataset/aptos-2019-blindness-detection-competition-dataset)  
10. APTOS-2019 dataset \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/datasets/mariaherrerot/aptos2019](https://www.kaggle.com/datasets/mariaherrerot/aptos2019)  
11. Robust Five-Class and binary Diabetic Retinopathy Classification Using Transfer Learning and Data Augmentation \- arXiv, accessed on November 30, 2025, [https://arxiv.org/html/2507.17121v1](https://arxiv.org/html/2507.17121v1)  
12. A. Dataset Detailed Information \- arXiv, accessed on November 30, 2025, [https://arxiv.org/html/2410.11428v1](https://arxiv.org/html/2410.11428v1)  
13. ctmedtech/EYEPACS · Datasets at Hugging Face, accessed on November 30, 2025, [https://huggingface.co/datasets/ctmedtech/EYEPACS](https://huggingface.co/datasets/ctmedtech/EYEPACS)  
14. DR Class Distribution in the Kaggle EYEPACs and APTOS Dataset \- ResearchGate, accessed on November 30, 2025, [https://www.researchgate.net/figure/DR-Class-Distribution-in-the-Kaggle-EYEPACs-and-APTOS-Dataset\_tbl1\_353713402](https://www.researchgate.net/figure/DR-Class-Distribution-in-the-Kaggle-EYEPACs-and-APTOS-Dataset_tbl1_353713402)  
15. Transfer Learning based Classification of Diabetic Retinopathy on the Kaggle EyePACS dataset \- Coventry University, accessed on November 30, 2025, [https://pureportal.coventry.ac.uk/en/publications/transfer-learning-based-classification-of-diabetic-retinopathy-on/](https://pureportal.coventry.ac.uk/en/publications/transfer-learning-based-classification-of-diabetic-retinopathy-on/)  
16. Messidor-2 \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess)  
17. Messidor-2 \- ADCIS, accessed on November 30, 2025, [https://www.adcis.net/en/third-party/messidor2/](https://www.adcis.net/en/third-party/messidor2/)  
18. Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research \- MDPI, accessed on November 30, 2025, [https://www.mdpi.com/2306-5729/3/3/25](https://www.mdpi.com/2306-5729/3/3/25)  
19. Diabetic Retinopathy Detection using PyTorch \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/code/balajiai/diabetic-retinopathy-detection-using-pytorch](https://www.kaggle.com/code/balajiai/diabetic-retinopathy-detection-using-pytorch)  
20. Diabetic Retinopathy Severity Classification Using Data Fusion and Ensemble Transfer Learning \- Scirp.org., accessed on November 30, 2025, [https://www.scirp.org/journal/paperinformation?paperid=138733](https://www.scirp.org/journal/paperinformation?paperid=138733)  
21. Project-MONAI/tutorials \- GitHub, accessed on November 30, 2025, [https://github.com/Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)  
22. Applying Ben's Preprocessing \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/code/banzaibanzer/applying-ben-s-preprocessing](https://www.kaggle.com/code/banzaibanzer/applying-ben-s-preprocessing)  
23. Result of 2nd Scenario with Ben Graham Preprocessing (a) Graph of 2nd... | Download Scientific Diagram \- ResearchGate, accessed on November 30, 2025, [https://www.researchgate.net/figure/Result-of-2nd-Scenario-with-Ben-Graham-Preprocessing-a-Graph-of-2nd-Scenario-Accuracy\_fig4\_361967953](https://www.researchgate.net/figure/Result-of-2nd-Scenario-with-Ben-Graham-Preprocessing-a-Graph-of-2nd-Scenario-Accuracy_fig4_361967953)  
24. Early Detection of Diabetic Retinopathy Using Pretrained Models: A Focus on Initial Stages, accessed on November 30, 2025, [https://ieeexplore.ieee.org/document/10730746/](https://ieeexplore.ieee.org/document/10730746/)  
25. A Novel Deep Learning Framework for Diabetic Retinopathy Detection Integrating Ben Graham and CLAHE Preprocessing \- NORMA@NCI Library, accessed on November 30, 2025, [https://norma.ncirl.ie/8643/1/namratashrishailtarade.pdf](https://norma.ncirl.ie/8643/1/namratashrishailtarade.pdf)  
26. EfficientNet for Diabetic Retinopathy: Healthcare ML Models \- Activeloop, accessed on November 30, 2025, [https://www.activeloop.ai/resources/efficient-net-for-diabetic-retinopathy-healthcare-ml-models/](https://www.activeloop.ai/resources/efficient-net-for-diabetic-retinopathy-healthcare-ml-models/)  
27. A Regression-Based Approach to Diabetic Retinopathy Diagnosis Using Efficientnet \- PMC, accessed on November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9955015/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9955015/)  
28. Diabetic Retinopathy Detection From Fundus Images Using Multi-Tasking Model With EfficientNet B5, accessed on November 30, 2025, [https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf\_icacc2022\_03027.pdf](https://www.itm-conferences.org/articles/itmconf/pdf/2022/04/itmconf_icacc2022_03027.pdf)  
29. Interpretable Deep Learning for Diabetic Retinopathy: A Comparative Study of CNN, ViT, and Hybrid Architectures \- ResearchGate, accessed on November 30, 2025, [https://www.researchgate.net/publication/391672133\_Interpretable\_Deep\_Learning\_for\_Diabetic\_Retinopathy\_A\_Comparative\_Study\_of\_CNN\_ViT\_and\_Hybrid\_Architectures](https://www.researchgate.net/publication/391672133_Interpretable_Deep_Learning_for_Diabetic_Retinopathy_A_Comparative_Study_of_CNN_ViT_and_Hybrid_Architectures)  
30. Interpretable Deep Learning for Diabetic Retinopathy: A Comparative Study of CNN, ViT, and Hybrid Architectures \- MDPI, accessed on November 30, 2025, [https://www.mdpi.com/2073-431X/14/5/187](https://www.mdpi.com/2073-431X/14/5/187)  
31. 1st place solution summary | Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary)  
32. Understanding The Metric: Quadratic Weighted Kappa (QWK) \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa](https://www.kaggle.com/code/carlolepelaars/understanding-the-metric-quadratic-weighted-kappa)  
33. Quadratic Weighted Kappa (QWK) Metric and How to Optimize It | by Anil Ozturk \- Medium, accessed on November 30, 2025, [https://medium.com/@nlztrk/quadratic-weighted-kappa-qwk-metric-and-how-to-optimize-it-062cc9121baa](https://medium.com/@nlztrk/quadratic-weighted-kappa-qwk-metric-and-how-to-optimize-it-062cc9121baa)  
34. How can I specify a loss function to be quadratic weighted kappa in Keras? \- Stack Overflow, accessed on November 30, 2025, [https://stackoverflow.com/questions/54831044/how-can-i-specify-a-loss-function-to-be-quadratic-weighted-kappa-in-keras](https://stackoverflow.com/questions/54831044/how-can-i-specify-a-loss-function-to-be-quadratic-weighted-kappa-in-keras)  
35. Preserving Ordinality in Diabetic Retinopathy Grading through a Distribution-Based Loss Function | OpenReview, accessed on November 30, 2025, [https://openreview.net/forum?id=TFQYxIUglD\&referrer=%5Bthe%20profile%20of%20Soufyan%20Lakbir%5D(%2Fprofile%3Fid%3D\~Soufyan\_Lakbir1)](https://openreview.net/forum?id=TFQYxIUglD&referrer=%5Bthe+profile+of+Soufyan+Lakbir%5D\(/profile?id%3D~Soufyan_Lakbir1\))  
36. EthanRosenthal/spacecutter: Ordinal regression models in PyTorch \- GitHub, accessed on November 30, 2025, [https://github.com/EthanRosenthal/spacecutter](https://github.com/EthanRosenthal/spacecutter)  
37. How to Perform Ordinal Regression / Classification in PyTorch | Towards Data Science, accessed on November 30, 2025, [https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/](https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/)  
38. arXiv:2403.15647v1 \[cs.CV\] 22 Mar 2024, accessed on November 30, 2025, [https://arxiv.org/pdf/2403.15647](https://arxiv.org/pdf/2403.15647)  
39. Towards a Transparent and Interpretable AI Model for Medical Image Classifications \- arXiv, accessed on November 30, 2025, [https://arxiv.org/html/2509.16685v1](https://arxiv.org/html/2509.16685v1)  
40. A Study on the Interpretability of Diabetic Retinopathy Diagnostic Models \- PMC, accessed on November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12649958/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12649958/)  
41. Enhanced SegNet with Integrated Grad-CAM for Interpretable Retinal Layer Segmentation in OCT Images \- arXiv, accessed on November 30, 2025, [https://arxiv.org/html/2509.07795v1](https://arxiv.org/html/2509.07795v1)  
42. Interpreting Deep Neural Networks in Diabetic Retinopathy Grading: A Comparison with Human Decision Criteria \- NIH, accessed on November 30, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12472159/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12472159/)  
43. A Comparative Survey of PyTorch vs TensorFlow for Deep Learning: Usability, Performance, and Deployment Trade-offs \- arXiv, accessed on November 30, 2025, [https://arxiv.org/html/2508.04035v1](https://arxiv.org/html/2508.04035v1)  
44. Galucoma Detection \- Pytorch/MONAI \- Kaggle, accessed on November 30, 2025, [https://www.kaggle.com/code/rasoulisaeid/galucoma-detection-pytorch-monai](https://www.kaggle.com/code/rasoulisaeid/galucoma-detection-pytorch-monai)  
45. Config management for deep learning : r/Python \- Reddit, accessed on November 30, 2025, [https://www.reddit.com/r/Python/comments/11o5a6m/config\_management\_for\_deep\_learning/](https://www.reddit.com/r/Python/comments/11o5a6m/config_management_for_deep_learning/)  
46. How to Configure Experiments With Hydra \- From an ML Enginner Perspective, accessed on November 30, 2025, [https://hackernoon.com/how-to-configure-experiments-with-hydra-from-an-ml-enginner-perspective](https://hackernoon.com/how-to-configure-experiments-with-hydra-from-an-ml-enginner-perspective)  
47. ML Experiment Tracking Tools: Comprehensive Comparison | DagsHub, accessed on November 30, 2025, [https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/](https://dagshub.com/blog/best-8-experiment-tracking-tools-for-machine-learning-2023/)  
48. MLflow vs Weights & Biases vs ZenML: What's the Difference?, accessed on November 30, 2025, [https://www.zenml.io/blog/mlflow-vs-weights-and-biases](https://www.zenml.io/blog/mlflow-vs-weights-and-biases)  
49. Model optimizations \- ONNX Runtime, accessed on November 30, 2025, [https://onnxruntime.ai/docs/performance/model-optimizations/](https://onnxruntime.ai/docs/performance/model-optimizations/)  
50. Deployment Options \- KodeKloud Notes, accessed on November 30, 2025, [https://notes.kodekloud.com/docs/PyTorch/Model-Deployment-and-Inference/Deployment-Options](https://notes.kodekloud.com/docs/PyTorch/Model-Deployment-and-Inference/Deployment-Options)  
51. Optimizing PyTorch Model Serving at Scale with TorchServe | UpStart Commerce, accessed on November 30, 2025, [https://upstartcommerce.com/optimizing-pytorch-model-serving-at-scale-with-torchserve/](https://upstartcommerce.com/optimizing-pytorch-model-serving-at-scale-with-torchserve/)