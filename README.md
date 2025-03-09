# Brazilian E-Commerce Dashboard ✨

## Setup Environment - Anaconda
```
conda create --name ecom-dashboard python=3.9
conda activate ecom-dashboard
pip install -r requirements.txt
```

## Setup Environment - Virtual Environment (venv)
```
python -m venv env
source env/bin/activate
env\Scripts\activate
pip install -r requirements.txt
```

## Setup Environment - Pipenv (Alternatif)
```
mkdir ecommerce_dashboard
cd ecommerce_dashboard
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Jalankan Aplikasi Streamlit
```
streamlit run dashboard/dashboard.py