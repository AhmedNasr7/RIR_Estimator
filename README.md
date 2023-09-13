# Anatomical-Context-Loss

## How to Run

### Method 1: Using Docker Compose

1. Make sure you have Docker and Docker Compose installed on your system.

2. Clone the repository:

```
git clone https://github.com/AhmedNasr7/RIR_Estimator
```

3. Navigate to the project directory:

```
cd RIR_Estimator
```

4. Run the application using Docker Compose:

```
docker-compose up
```


5. Run Training

```
python3 train.py
```


### Method 2: Using a Virtual Environment

1. Clone the repository:


```
git clone https://github.com/AhmedNasr7/RIR_Estimator
```


2. Navigate to the project directory:

```
cd RIR_Estimator
```


3. Create a virtual environment (optional but recommended):

```
python3 -m venv env
```


4. Activate the virtual environment:

- On Windows:
  ```
  .\env\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source env/bin/activate
  ```

5. Install the required dependencies:

```
python3 -m pip install -r requirements.txt
```


6. Run Training

```
python3 train.py
```






