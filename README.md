# Pattern Is All You Need: Predicting Atomic Properties from Aggregate Molecular Data

This repository contains the code and resources for the thesis project "Pattern Is All You Need," developed at the Clean Energy Lab, University of Toronto. This research addresses a weak supervision problem in materials science, focusing on the prediction of individual atomic properties from aggregate molecular data using neural networks.

## Table of Contents

  - [Introduction](https://www.google.com/search?q=%23introduction)
  - [Problem Statement](https://www.google.com/search?q=%23problem-statement)
  - [Approach](https://www.google.com/search?q=%23approach)
  - [Key Contributions](https://www.google.com/search?q=%23key-contributions)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Technologies Used](https://www.google.com/search?q=%23technologies-used)
  - [Author](https://www.google.com/search?q=%23author)
  - [License](https://www.google.com/search?q=%23license)

## Introduction

In many scientific domains, particularly in materials science and chemistry, obtaining direct supervision for individual atomic properties can be challenging or impossible. Instead, data often comes in an aggregate form, representing properties of an entire molecule or system. This project explores the application of neural networks to infer individual atomic contributions from such aggregate molecular data, tackling the inherent weak supervision problem.

## Problem Statement

The core challenge lies in learning atomic contributions when direct atomic-level supervision is unavailable. Given only molecular-level properties, how can a neural network effectively disentangle and predict the properties of constituent atoms? This requires the development of models that can implicitly learn the relationship between atomic features and their collective impact on molecular properties.

## Approach

This project emphasizes the development of position-invariant and shared-weight neural architectures. These architectures are designed to be robust to the spatial arrangement of atoms within a molecule and to apply the same learning principles across different atoms, reflecting the underlying physical principles. The research investigates how architectural design and inductive bias significantly influence the model's ability to generalize to unseen data and scale to larger, more complex molecular systems.

## Key Contributions

  - **Development of Novel Neural Architectures:** Implementation and evaluation of neural network architectures specifically designed for weak supervision in atomic property prediction.
  - **Investigation of Inductive Bias:** Analysis of how different architectural choices and inductive biases impact the model's generalization capabilities and scalability.
  - **Addressing Weak Supervision:** A practical demonstration of using neural networks to infer fine-grained properties from coarse-grained, aggregate data.
  - **Application in Clean Energy Research:** Contribution to the field of clean energy by developing computational tools that can aid in the understanding and design of materials at the atomic level.

## Project Structure

The repository is organized into the following main directories and files:

  - `main/`: Contains the primary source code for the neural network models and training scripts.
  - `numpy_arrays/`: Stores NumPy arrays, containing inputs/outputs.
  - `presentations/`: Contains presentation materials related to the project.
  - `results/`: Stores the output of model training, evaluation, and analysis.
  - `sumenv/`: Contains venv related files.

## Technologies Used

The project primarily utilizes the following programming languages and frameworks:

  - **Python** (70.8%)
  - **C++** (21.7%)
  - **NASL** (4.7%)
  - **Cython** (1.3%)
  - **C** (1.1%)
  - **CUDA** (0.2%)

## Author

Kanishk Yadav
University of Toronto & BITS Pilani
December 2023
