@echo off
echo Creating Stock Sentiment Tracker Project Structure...
echo.

:: Create main directories
mkdir app
mkdir scripts
mkdir tests
mkdir notebooks
mkdir data
mkdir logs
mkdir ml_models
mkdir config
mkdir dashboard
mkdir dashboard\static
mkdir dashboard\templates
mkdir app\features
mkdir app\models
mkdir app\training
mkdir app\inference
mkdir app\evaluation
mkdir data\raw
mkdir data\processed
mkdir data\historical
mkdir logs\errors
mkdir logs\user_actions
mkdir logs\model_logs
mkdir ml_models\trained
mkdir ml_models\experiments

echo Directories created successfully!
echo.

:: Create ROOT files
echo # Stock Sentiment Tracker Project > README.md
echo. > requirements.txt
echo. > .env.example
echo. > .gitignore
echo. > Dockerfile
echo. > docker-compose.yml
echo. > setup.py
echo. > LICENSE
echo. > .dockerignore

:: Create APP files
cd app
echo. > __init__.py
echo. > main.py
echo. > config.py
echo. > data_collector.py
echo. > sentiment_analyzer.py
echo. > model_trainer.py
echo. > utils.py
echo. > feature_engineer.py
echo. > ensemble_model.py
echo. > lstm_model.py
echo. > predictor.py

:: Create features files
cd features
echo. > __init__.py
echo. > technical_features.py
echo. > sentiment_features.py
echo. > market_features.py
cd..

:: Create models files
cd models
echo. > __init__.py
echo. > xgb_model.py
echo. > finbert_sentiment.py
cd..

:: Create training files
cd training
echo. > __init__.py
echo. > trainer.py
echo. > validator.py
echo. > hyperparameter_tuner.py
cd..

:: Create inference files
cd inference
echo. > __init__.py
echo. > realtime_predictor.py
echo. > confidence_calibrator.py
cd..

:: Create evaluation files
cd evaluation
echo. > __init__.py
echo. > metrics.py
echo. > backtester.py
echo. > shap_analyzer.py
cd..

:: Go back to root
cd..

:: Create scripts directory files
cd scripts
echo. > __init__.py
echo. > collect_data.py
echo. > train_model.py
echo. > run_pipeline.py
echo. > download_sample_data.py
echo. > realtime_predictor.py
echo. > cleanup.py
echo. > deploy.py

:: Create api subdirectory
mkdir api
cd api
echo. > __init__.py
echo. > binance_api.py
echo. > news_api.py
echo. > twitter_api.py
cd..

:: Create utils subdirectory
mkdir utils
cd utils
echo. > __init__.py
echo. > data_cleaner.py
echo. > logger.py
echo. > email_notifier.py
cd..

:: Create deployment subdirectory
mkdir deployment
cd deployment
echo. > __init__.py
echo. > docker_build.py
echo. > cloud_deploy.py
cd..

cd..

:: Create tests files
cd tests
echo. > __init__.py
echo. > test_data_collection.py
echo. > test_sentiment.py
echo. > test_models.py
echo. > test_features.py
echo. > test_utils.py
echo. > test_integration.py
echo. > conftest.py
cd..

:: Create notebooks files
cd notebooks
echo. > 01_eda.ipynb
echo. > 02_feature_engineering.ipynb
echo. > 03_model_experiments.ipynb
echo. > 04_backtesting.ipynb
echo. > 05_sentiment_analysis.ipynb
echo. > 06_production_pipeline.ipynb
cd..

:: Create config files
cd config
echo. > __init__.py
echo. > settings.py
echo. > api_keys.py
echo. > database_config.py
echo. > model_config.yaml
echo. > logging_config.yaml
cd..

:: Create dashboard files
cd dashboard
echo. > __init__.py
echo. > app.py
echo. > charts.py
echo. > components.py
echo. > layout.py

cd static
echo. > style.css
echo. > custom.js
echo. > favicon.ico
cd..

cd templates
echo. > base.html
echo. > index.html
echo. > dashboard.html
echo. > settings.html
cd..

cd..

echo.
echo Project structure created successfully!
echo.
echo Total directories created: 
dir /ad /s | find "Directory(s)"
echo.
echo Total files created:
dir /a-d /s | find "File(s)"
echo.
pause