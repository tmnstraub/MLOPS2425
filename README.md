## Predicting Wine Price MlOps Pipeline

This project implements a modular MLOps pipeline for preparing, processing, and modeling a wine dataset using Kedro. The pipeline is designed to replicate real-world production workflows, with each pipeline covering a specific stage of the machine learning lifecycle:

* **Ingestion Pipeline** (`ingestion_pipeline`):
  Loads the raw dataset into the project environment, preparing it for downstream processing.
* **Data Quality Pipeline** (`data_quality_pipeline`):
  Runs data unit tests on the ingested data, checking schema consistency, types, duplicates, and expected distributions. Outputs reports and visualizations to assess data quality before further processing.
* **Preprocessing Pipeline** (`preprocessing_pipeline`):
  Applies data cleaning and preprocessing steps separately to both **batch** and **train** datasets, including:
  * Missing value handling.
  * Normalization.
  * General transformations.
* **Feature Engineering Pipeline** (`feature_engineering_pipeline`):
  Runs on both ****batch** and** **train** datasets. Key tasks include:
  * Creating new features.
  * Storing engineered datasets for reproducibility.
  * Dynamically dropping columns with high correlation, guided by insights from the Streamlit data catalog (Cramér’s V categorical correlation heatmaps).
  * Managing columns to drop via  `features_to_drop` in `parameters.yml`.
  * Applying one-hot encoding where required.
* **Train/Validation Split Pipeline** (`train_validation_split_pipeline`):
  Splits the train dataset (both one-hot encoded and non-encoded versions) into training and validation subsets, enabling robust model selection and evaluation.
* **Model Selection Pipeline** (`model_selection_pipeline`):
  Searches and compares candidate models, evaluating them on the train/validation data with defined metrics.
* **Feature Selection Pipeline** (`feature_selection_pipeline`):
  Uses the best-performing model to select optimal features through methods like tree-based importance or Recursive Feature Elimination (RFE).
* **Model Train Pipeline** (`model_train_pipeline`):
  Trains the final model with the best features selected, using the chosen model and hyperparameters.
* **Model Predict Pipeline** (`model_predict_pipeline`):
  Applies the trained model to the **batch** dataset, generating predictions on new or unseen data.

---

### Design highlights

* **Parallel data preparation for different model types** :
  Supports both one-hot and non-one-hot paths, enabling flexible experiments with models like CatBoost (native categorical support) or others requiring numeric inputs.
* **Dynamic feature selection** :
  Columns to drop are configured in `parameters.yml` and can be adjusted without code changes.
* **Reproducibility and transparency** :
  Engineered datasets and feature selections are stored, while data quality is visualized and reported.
* **Streamlit integration** :
  Correlation heatmaps and dataset exploration are available in a Streamlit data catalog interface.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

---

## How to install dependencies

This project requires Python 3.9.23 and a number of packages. You can set up the project environment as follows:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Important notes

- The project has been tested with Python 3.9.23
- The `requirements.txt` file contains all necessary packages with compatible versions
- For feature store functionality, specific package versions (hsfs, hopsworks, etc.) are required and included

The feature store integration requires specific versions of these packages:

- `hsfs==3.7.9`
- `hopsworks==4.2.6`
- `hopsworks-aiomysql==0.2.1`

If you encounter issues with the feature store upload, ensure these versions are installed correctly.

---



## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter

To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab

To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython

And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)


## Docker

To create a docker file based on the main run:

```
docker build -t wine-project:2.1 .
```

To load the .tar file of the created docker image run: 

```
docker load -i wine-project_2.1.tar
```