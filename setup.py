from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="webinar-reframer",
    version="1.0.0",
    author="Yash",
    author_email="pansareyash9740@gmail.com",
    description="Automatically reframe horizontal webinar videos to vertical 9:16 format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YashPansare31/-mediapipe-autoflip-reframe.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Content Creators",
        "Topic :: Multimedia :: Video :: Conversion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "enhanced": [
            "scikit-image>=0.21.0",
            "scipy>=1.11.0",
        ],
        "gpu": [
            "onnxruntime-gpu>=1.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "webinar-reframer=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.pbtxt", "data/samples/*.mp4"],
    },
)