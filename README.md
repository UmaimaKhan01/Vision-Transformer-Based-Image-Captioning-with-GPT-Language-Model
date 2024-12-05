# Vision-Transformer-Based-Image-Captioning-with-GPT-Language-Model

Our approach combines the state-of-the-art performance of transformers in both computer vision and natural language processing domains to create descriptive and contextually relevant captions for a wide range of images.
![image](https://github.com/user-attachments/assets/2904d6f6-3ff4-4a88-8605-dc9ce7b0fa74)
![image](https://github.com/user-attachments/assets/5477bc90-1a2e-40d4-acd7-661f1c2aa36c)

### Steps to Run the Image Captioning Project

Follow these steps to set up and run the project for generating captions using the COCO 2017 dataset:

---

#### 1. **Clone the Repository**
Download the project code from the repository to your local system:
```bash
git clone https://github.com/your-username/image-captioning-project.git
cd image-captioning-project
```

---

#### 2. **Install Dependencies**
Install the required Python libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

#### 3. **Prepare the Dataset**
Ensure you have access to the COCO 2017 dataset. The dataset will be downloaded automatically if it’s not already available when running the script.

---

#### 4. **Fine-Tune the Model**
Customize the pre-trained Vision Transformer model on the COCO 2017 dataset:
1. Run the fine-tuning script:
   ```bash
   python fine_tuning.py
   ```
2. The model will be trained and saved in the `./results` directory.

---

#### 5. **Start the Image Captioning Server**
Launch the API server to generate captions:
```bash
python image_captioning_server.py
```
The server will start at `http://127.0.0.1:8000`.

---

#### 6. **Test the Server**
Send a test request to the server:
1. Use a tool like [Postman](https://www.postman.com/) or your browser to access `http://127.0.0.1:8000/docs` for API documentation.
2. Alternatively, use the provided client script:
   ```bash
   python client.py
   ```

---

#### 7. **Generate Captions**
Send an image to the server to generate captions:
1. Update the `client.py` script with the image path or URL:
   ```python
   payload = {"image_path": "path_to_your_image.jpg"}
   ```
2. Run the script:
   ```bash
   python client.py
   ```

---

#### Notes:
- Ensure Python (3.8 or later) and required libraries are installed.
- If the server doesn’t start, verify that no other process is using port `8000` or change the port in `image_captioning_server.py`.

---

This is a straightforward guide to get your project up and running. Copy this into your README file for ease of use.
