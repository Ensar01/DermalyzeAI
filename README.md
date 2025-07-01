# 🧬 Dermalyze - Skin Disease Classification 

<p><img src="https://github.com/Ensar01/DermalyzeAI/blob/main/static/pics/Logo.png"></p>

Dermalyze uses a fine-tuned Vision Transformer (ViT) architecture trained on the DermNet dataset. The model analyzes skin images to identify conditions such as acne, psoriasis, dermatitis, and more. It enables fast and preliminary assessment of potential dermatological conditions, with the ability to recognize 23 different skin diseases.

## 📷 Model Overview

- **Architecture:** ViT `vit-base-patch16-224-in21k`
- **Input size:** 224x224 RGB images
- **Fine-tuned on:** DermNet dataset (23 skin disease categories)

## ⚙️ Training Setup

| Setting            | Value             |
|--------------------|-------------------|
| Optimizer          | AdamW             |
| Learning Rate      | 0.00002           |
| Loss Function      | CrossEntropyLoss  |
| Batch Size         | 16                |
| Epochs             | 10                |
| Early Stopping     | Patience = 3      |

## 🧪 Example Prediction
```bash 
Predicted class: Acne and Rosacea Photos
Confidence Score: 96.97%
```

## 🔍 Limitations

- Dataset may be biased toward lighter skin tones
- Model is not clinically validated
- Should not be used as a substitute for medical diagnosis
