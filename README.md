# How to Run the Python File

Follow these steps to run the Python file in this repository:

## 1. Create a Virtual Environment

- To isolate dependencies, create a virtual environment:
```bash
python -m venv venv
```

## 2. Activate the Virtual Environment

- On **Windows**: 
```bash
venv\Scripts\activate
```
- On **Mac/Linux**: 
```bash
source venv/bin/activate
```

## 3. Install Dependencies

- Use the `requirements.txt` file to install dependencies:
```bash
pip install -r requirements.txt

``` 
- Then, you should be able to run the script using:
```bash
python langchain_ui_testing.py

```

**Note**: This documents assumes you have the models in the python file locally installed using **Ollama**. 
