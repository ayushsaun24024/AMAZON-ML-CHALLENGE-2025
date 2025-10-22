# 🛒 Amazon ML Challenge 2025: Predicting Product Prices from Catalog Data

**Team:** Manthan Minds  
**Members:** Ayush Saun, Jay Sharma, Manish Vardhan, Abhay Pratap Singh  
**Rank:** 304  
**Score:** 46.0857  

This repository showcases the work of **Manthan Minds** in the Amazon ML Challenge 2025. Our goal was to predict product prices using the catalog data provided by Amazon, which included **both text descriptions and product images**. Participants were free to use either or both modalities to improve predictions.

---

## 📌 Problem Statement

Predict the price of a product based on its catalog information. Each product had:

- `sample_id`: Unique identifier  
- `catalog_content`: Product title, description, and other details  
- `image_link`: Public URL of the product image  
- `price`: Target variable (available only in training data)  

The challenge tested our ability to analyze both textual and visual data and build a robust predictive model.

---

## 🔍 Our Approach

We focused on a **practical and effective pipeline** that combined data cleaning, feature extraction, and model optimization:

1. **Data Preprocessing** 🧹  
   - Cleaned and standardized product text.  
   - Truncated long descriptions to focus on the most relevant information.  
   - Removed extreme outlier prices to help the model learn realistic patterns.  
   - Applied transformations to normalize price distributions.  

2. **Feature Extraction** 🧠  
   - Leveraged the textual descriptions to capture product context.  
   - Images were available, but we focused primarily on text features for this iteration, keeping the approach flexible for future multimodal enhancements.  

3. **Modeling** 🏗️  
   - Used transformer-based models to understand the semantics of product descriptions.  
   - Added a regression layer on top of the transformer to predict continuous price values.  
   - Trained and validated the model iteratively, tracking performance and saving the best models.  

4. **Evaluation & Optimization** 📊  
   - Used SMAPE (Symmetric Mean Absolute Percentage Error) as the main evaluation metric.  
   - Normalized and transformed predictions back to the original price scale for accuracy.  
   - Applied early stopping to prevent overfitting and ensure generalization.  

5. **Results** 🏆  
   - Achieved a **SMAPE score of 46.0857**  
   - Secured **Rank 304** in the competition  

---

## 💡 Reflections

- Carefully preprocessing text and handling outliers greatly improved model performance.  
- Transforming prices (e.g., log scaling) helped the model learn better.  
- Even without using image features directly, text-based transformers captured much of the price signal.  
- Collaboration among team members was key to iterating quickly and improving results.  

---

## 🚀 Future Directions

- Integrate product images to build a multimodal predictor.  
- Explore ensemble models combining multiple architectures.  
- Experiment with advanced NLP and CV techniques for better feature extraction.  
- Fine-tune preprocessing steps and hyperparameters for potential improvements.  

---

This project demonstrates how thoughtful data handling, combined with modern machine learning techniques, can solve real-world e-commerce pricing problems effectively. 🎉
