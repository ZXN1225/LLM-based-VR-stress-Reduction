# LLM-based-VR-stress-Reduction
A LLM based 360 panorama and music generation pipeline.

1.Put all your environment configuration (API key .etc.)into the .env file.

2.Put your panorama datasets in PictureData folder.

3.Run Extraction_Database.py and Mu_Extraction.py to get database and model for FAED calculation.

3.1 If you already have a LoRA, put it in models/lora folder.

3.2 If you want to train your own LoRA based on your datasets, run Convert_LoRA_Dataset.py to get caption and then train your own LoRA with the caption and tools (such as Kohyass).

4.Run AgentAPP.py and then you can open the localhost link to run the pipeline.

5.Config your Unity to connect and run in VR environment.
