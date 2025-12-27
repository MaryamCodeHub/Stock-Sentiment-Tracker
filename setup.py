from setuptools import setup, find_packages

setup(
    name="stock-sentiment-tracker",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time stock price and sentiment prediction system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-sentiment-tracker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "stock-tracker=app.main:main",
            "stock-train=scripts.train_model:main",
            "stock-collect=scripts.collect_data:main",
        ],
    },
) 
