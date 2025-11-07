Production-Ready ML Solution Design for Finance MLE Assessment
Project Overview and Goals

The goal is to build an end-to-end machine learning framework (repository name ml-sys) that enables data scientists to gather and transform data, rapidly build/iterate models, deploy them, and measure their impact. We focus on a customer conversion prediction use-case: given HubSpot prospect usage data and company info, predict which prospects will convert to paying customers. The solution must be implemented in pure Python (no heavy orchestration frameworks) and adhere to strong software engineering principles expected of a senior engineer. This means a modular, maintainable codebase with clear interfaces, reusable utilities, and CLI-style entry points for ease of use. We emphasize reproducibility (config-driven runs, fixed random seeds) and clarity (type hints, docstrings, well-organized code) as outlined in the assessment guidelines.

Key design objectives include:

End-to-End Pipeline: A robust pipeline that covers data loading, preprocessing, class rebalancing, model training, evaluation, and deployment.

Modular Code & Clean Interfaces: Separation of concerns (data handling, modeling, etc.) with clear function/class interfaces and reusability.

Reproducibility: Config files or parameters control experiments; environment lock files or requirements ensure consistent setups. Artifacts (models, metrics) are saved for later reuse.

Production Readiness: Integration of CI/CD for linting, formatting, and testing, model versioning for traceability, and an API for real-time predictions.

Usability: Simple CLI scripts (train.py, predict.py) to run tasks, and comprehensive documentation (README, comments) so that team members can easily use and extend the framework.

Below we describe the pipeline and system design in detail, covering each stage from data ingestion to deployment, as well as project structure and DevOps considerations.

Data Ingestion & Preprocessing Pipeline

Data Sources: The framework will load raw data from the provided CSV files: customers.csv (paying customers), noncustomers.csv (prospect companies), and usage_actions.csv (product usage logs). Each company has a unique ID key present across these files. Important fields include company attributes (industry, employee range, Alexa rank, etc.), whether and when they converted (CLOSEDATE for customers), and time-stamped usage activity counts (actions on contacts, deals, emails, etc.). We assume the data is a static snapshot (no live updates) taken right after the latest timestamp in usage_actions.csv.

Data Loading: A module in src/data/ (e.g. data_loader.py) will handle reading these CSVs (using pandas). It will join and transform them into a usable feature set and label:

We create a master DataFrame of companies, using ID as the primary key. Companies present in customers.csv are labeled as converted = 1, and those only in noncustomers.csv are converted = 0 (the target variable). This aligns with the definition of “conversion” as moving from a free prospect to a paying customer.

We merge usage data by aggregating usage_actions.csv per company. For simplicity, we can derive features such as total actions (or last N days' actions) for each company across various categories (CRM contacts, deals, emails, etc.). More sophisticated temporal features (like trends or recency of usage) could be engineered, but initial implementation may use cumulative or latest values of these usage metrics for each company.

Any needed data cleaning is performed: e.g., handling missing values (if any industry or rank is missing, impute or flag it), converting date fields to datetime (CLOSEDATE) if needed for feature engineering (like tenure since conversion, though for non-customers this might be null).

Feature Encoding: The dataset includes categorical features such as INDUSTRY and EMPLOYEE_RANGE. We will encode these for modeling. Given our model choice (XGBoost), one approach is to use ordinal or one-hot encoding via scikit-learn (e.g., OrdinalEncoder for industry). These encoders will be fitted on the training set only to prevent leakage. The encoding objects should be saved as artifacts for reuse in inference.

Train-Test Split: We split the prepared data into training and test sets (for evaluation) using an 80/20 stratified split (to preserve class balance). Stratification ensures the minority class (converted customers) is represented in both sets similarly
medium.com
. We also set aside a part of training data for validation if needed (or use cross-validation) to tune hyperparameters, though a simple approach may use a single split given time constraints. All splitting occurs before any resampling to avoid information bleed.

Addressing Class Imbalance – SMOTENC: Since typically only a small fraction of prospects convert to customers, the classes will be imbalanced (many more noncustomers than customers). To rebalance, we apply SMOTE-NC (SMOTENC) from the imbalanced-learn library on the training data only
medium.com
. SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features) generates synthetic samples for the minority class by interpolating between existing minority samples, with special handling for categorical features
medium.com
. In essence, it creates new "customer" examples: numeric features are averaged with nearest neighbors, while categorical features for a synthetic sample are chosen by majority vote among neighbors
medium.com
. This helps the model not bias toward the majority class, without requiring us to drop data. We ensure SMOTENC is only fit on the training split (the algorithm internally uses KNN on training minority samples) and do not oversample the test set, to maintain a fair evaluation
medium.com
. The SMOTENC class requires specifying which feature indices are categorical; we will pass those (or use pandas dtypes to detect) so that one-hot encoding is handled internally if needed.

After oversampling, the training set will have a more balanced class distribution. We may optionally perform feature scaling (e.g., standardization) on numeric features, though tree-based models like XGBoost are less sensitive to scaling. If we do, that scaler would also be fit on train data only.

Reproducibility: All random processes (train/test split shuffling, SMOTENC oversampling, model training initialization) will use fixed random_state seeds for deterministic runs. Config files (e.g., a YAML in configs/data_config.yaml) can specify parameters like test split ratio, SMOTE oversampling ratio (default 'auto' to equalize classes), or any filtering logic, making preprocessing easily adjustable without code changes.

By the end of this stage, we have a prepared training set (oversampled) and a test set, ready for modeling. We also have saved preprocessing artifacts (encoders, possibly a scaler) that will be needed for inference.

Model Training and Evaluation

Model Choice - XGBoost: We select XGBoost (Extreme Gradient Boosting) as our classifier (using the Python xgboost library). XGBoost is well-suited for structured data and often achieves state-of-the-art results for classification with minimal tuning. It can handle large datasets efficiently via boosting and provides built-in handling to prevent overfitting (regularization, tree pruning). It was also highlighted in similar churn prediction scenarios as a robust choice
python.plainenglish.io
. Additionally, XGBoost can ingest our mix of features (which are all numeric after encoding) and handle non-linear relationships.

Training Procedure: The training process will be encapsulated in a script (e.g., scripts/train.py) which orchestrates the following steps:

Load Config: The script reads a config file (e.g., configs/train_config.yaml) or CLI args for hyperparameters (max_depth, learning_rate, etc.), number of training epochs/trees, and file paths.

Data Loading & Preprocessing: It calls the data module to load and preprocess data as described above. This yields X_train, y_train, X_test, y_test. (If a validation split is needed for early stopping, that can be carved out of X_train or using XGBoost’s built-in eval_set parameter).

Initialize Model: Create an XGBClassifier with specified hyperparams. For example, we might use xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=1) etc. If using oversampled data, we may not need scale_pos_weight (which otherwise helps with imbalance).

Train: Call model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='auc', early_stopping_rounds=10) for instance. This trains the model, using ROC-AUC on the test set for early stopping (to prevent overfitting). We ensure no leakage: the test set is only used for monitoring, not for training or oversampling.

Save Artifacts: After training, save the model to disk (e.g., using Python's pickle or joblib). For example, joblib.dump(model, "models/conversion_model_v1.0.0.pkl") to save the trained model (with a versioned filename). We also save the preprocessing objects: e.g., encoder.pkl for the industry/employee encoders if applicable, so the API can use the exact same transformations. All artifacts can reside in a models/ directory, possibly organized by timestamp or version.

Evaluation Metrics: The framework will compute both ROC-AUC and PR-AUC on the test set:

ROC-AUC (Receiver Operating Characteristic Area Under Curve) measures the model’s ability to rank positive vs negative instances across all classification thresholds. It is threshold-independent and useful for overall performance. We compute it by obtaining predicted probabilities y_pred_proba for the positive class and using roc_auc_score(y_test, y_pred_proba)
neptune.ai
.

PR-AUC (Precision-Recall AUC) focuses on the trade-off between precision and recall for the positive class. This is particularly informative for imbalanced datasets where we care about the minority (converted customers)
neptune.ai
. A high PR-AUC means when the model does predict a conversion, it’s usually correct (high precision) and it captures a large fraction of actual conversions (high recall). We compute PR-AUC using average_precision_score from scikit-learn (which actually computes the area under the precision-recall curve)
neptune.ai
.

By evaluating both, we ensure we assess model quality comprehensively. (ROC-AUC can sometimes be high even when the model’s precision on positives is low, so PR-AUC gives that perspective
neptune.ai
.) We might also record other metrics like F1-score, accuracy (though accuracy is less useful under imbalance), and perhaps a confusion matrix for clarity.

Results & Artifact Logging: The training script will log the metrics (print to console and/or save to a JSON results file). For example, after training:

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
pr_auc = average_precision_score(y_test, model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")


These metrics, along with training parameters, can also be saved to models/metrics_v1.0.0.json for record-keeping. If integrated with a tracking tool (optional), we could log to MLflow or similar, but to keep it simple we stick to lightweight logging.

We avoid any train/test leakage throughout (no oversampling or encoding fitted on test). The pipeline is reproducible: running train.py with the same config yields the same model and metrics (thanks to fixed seeds and saved data splits). With this setup, data scientists can tweak hyperparameters or feature processing in config and quickly re-run training to iterate on models (fulfilling the “rapidly build and iterate” objective).

Model Deployment and Inference API (FastAPI)

After training, we package the model for deployment with a FastAPI web service, enabling real-time predictions. The deployment component (in src/api/) will be a REST API that loads the trained model and exposes endpoints for inference. This addresses the “host and deploy models” requirement.

FastAPI App Setup: We create a FastAPI application (e.g., in src/api/app.py). On startup, the app will load the latest model artifact from disk. For example:

from fastapi import FastAPI
import joblib

app = FastAPI(title="Conversion Prediction API", version="1.0.0")

# Global variables to hold the model and encoders
model = None
encoder = None

@app.on_event("startup")
def load_model_artifacts():
    """Load model and encoder into memory at startup."""
    global model, encoder
    model = joblib.load("models/conversion_model_v1.0.0.pkl")
    encoder = joblib.load("models/industry_encoder.pkl")
    # ... load other preprocessing objects as needed


Here we set the API version as 1.0.0 (matching model version). We use FastAPI’s startup event to load artifacts once, rather than loading on each request (for efficiency)
python.plainenglish.io
. The global model will then be used inside request handlers. We include a health check endpoint as well:

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
}


This allows quick checks that the service is up and the model is ready
python.plainenglish.io
.

Request Schema: Define a Pydantic model for the input features JSON. For example, if our model uses features like ALEXA_RANK, INDUSTRY, EMPLOYEE_RANGE, ACTIONS_CRM_CONTACTS, ... etc., we create a class:

from pydantic import BaseModel

class CompanyFeatures(BaseModel):
    industry: str
    employee_range: str
    alexa_rank: int
    actions_crm_contacts: int
    actions_crm_deals: int
    # ... include all necessary features as per training


Using Pydantic ensures the request JSON is validated and parsed into a Python object (CompanyFeatures) automatically.

Prediction Endpoint: We implement a POST endpoint that takes a JSON payload of features and returns the model prediction. For example:

@app.post("/predict")
def predict_conversion(data: CompanyFeatures):
    # Prepare the input for model
    df = data.dict()  # convert to dict
    # Apply preprocessing: e.g., encode industry
    df["industry"] = encoder.transform([[df["industry"]]])[0][0]
    # Convert to DataFrame (XGBoost expects 2D input)
    X = pandas.DataFrame([df])
    # Get prediction probability for class "converted"
    proba = model.predict_proba(X)[0, 1]
    # Also return a binary prediction (Yes/No) based on threshold 0.5
    pred = "Yes" if proba >= 0.5 else "No"
    return {"conversion_probability": round(float(proba),4), "will_convert": pred}
}


This function takes the input, applies the same encoding as training (using the loaded encoder for industry, etc.), then uses model.predict_proba to get the probability of conversion (class 1). We return the probability (rounded for readability) and a simple yes/no prediction. We could also map the probability to a risk tier (low/medium/high) if useful, similar to how a churn model might classify risk
python.plainenglish.io
python.plainenglish.io
, but for conversion a probability is likely sufficient.

The FastAPI app automatically provides a docs UI (Swagger) at /docs thanks to Pydantic, which clearly communicates the API contract (fulfilling the requirement to communicate the API well). We also include basic error handling: if the model isn't loaded or an encoder fails (e.g., an unseen category), we return HTTP 400/500 with an error message (using FastAPI's HTTPException as seen in similar implementations
python.plainenglish.io
python.plainenglish.io
).

Running the API: We can serve this with Uvicorn. A CLI script scripts/serve.py might call:

uvicorn src.api.app:app --host 0.0.0.0 --port 8000


to start the server. (In production, this could be managed by a process manager or container orchestration, but for this design we assume manual or basic deployment.)

The API allows integration into a live system – e.g., the sales tool could call this /predict endpoint for a given prospect to get a conversion likelihood score. This deployment is lightweight and purely in Python. It can be containerized later if needed (FastAPI is easily dockerized), but containerization is optional here (the assessment mentioned it as one option for reproducibility but not strictly required).

Repository Structure and Code Organization

We organize the repository in a standard, production-oriented structure for clarity and scalability:

ml-sys/
├── src/
│   ├── data/              # data loading & transformation modules
│   │   └── loader.py      # functions to read CSVs and merge into features
│   ├── features/          # feature engineering and preprocessing
│   │   ├── preprocessing.py  # e.g., functions for encoding, SMOTENC pipeline
│   │   └── __init__.py
│   ├── models/            # modeling logic
│   │   ├── train_model.py    # training routine (could define a Trainer class or functions)
│   │   ├── evaluate.py       # evaluation metrics computations
│   │   └── predict_model.py  # batch prediction functions (if any)
│   ├── api/              # API application
│   │   └── app.py        # FastAPI app definition with endpoints
│   ├── utils/            # (optional) utility functions (logging, config parsing, etc.)
│   └── __init__.py       # makes src a package
├── scripts/
│   ├── train.py          # CLI entry: loads config and calls training
│   ├── evaluate.py       # CLI entry: (if separate evaluation run needed)
│   └── serve.py          # CLI entry: starts the FastAPI server (or could just instruct uvicorn)
├── configs/
│   ├── config.yaml       # master config including paths, hyperparams, etc.
│   └── logging.conf      # logging configuration (if using logging module)
├── tests/
│   ├── test_data_loader.py    # unit tests for data loading
│   ├── test_features.py       # tests for preprocessing (e.g., SMOTENC integration)
│   ├── test_model_training.py # tests for training (maybe using a small sample)
│   └── test_api.py            # tests for the FastAPI routes (could use TestClient)
├── models/
│   └── (this will contain saved model artifacts like .pkl files after training)
├── .github/workflows/
│   └── ci.yml            # GitHub Actions workflow for CI
├── requirements.txt      # Python dependencies (exact versions for reproducibility)
├── README.md             # Documentation for the project
└── pyproject.toml / setup.py   # (optional) if we make this an installable package


This layout separates different aspects of the ML system. The src/ directory can be made an installable package (e.g., via uv pip install -e . or more commonly uv sync for local development), which would allow us to use imports like from ml_sys.data.loader import load_data. While we may not publish it, structuring as a package enforces good organization. Notably:

Data modules handle raw data ingestion and basic transforms. They do not contain model code.

Feature modules might contain code to create new features or to apply SMOTENC balancing. For example, a function apply_smote(X, y, categorical_indices) -> X_res, y_res lives here.

Model modules encapsulate training and inference logic. For instance, a ModelTrainer class could coordinate training, and a predict_model.py might have a helper to load a model and run batch predictions on a file of new examples.

API is isolated, so it only concerns serving and uses the model module for any heavy lifting (e.g., it calls models.predict_model.load_model() and models.predict_model.predict() internally, rather than duplicating logic).

This modularization makes it easier for teammates to extend the framework. For example, adding a new data source would involve writing a new loader in src/data/ and plugging it in, without touching model or API code, adhering to open/closed principle. Common patterns (like data loading, model saving) are abstracted into utility functions or base classes so they can be reused and overridden if needed (the assessment expects abstraction of common patterns for future extensibility).

Coding Standards: All code is written with type hints for functions and classes, and includes docstrings explaining usage. This communicates the API of each component clearly to other developers. For example, def load_data(...) -> pd.DataFrame: with a docstring describing return values. We also include in-line comments for any tricky logic. The style will be made consistent by adhering to PEP8 and using automated formatters/linters (configured in CI as discussed below).

Configuration Management: Instead of hard-coding parameters, we use config files and/or CLI arguments. For instance, configs/config.yaml might contain:

data_paths:
  customers: "data/customers.csv"
  noncustomers: "data/noncustomers.csv"
  usage: "data/usage_actions.csv"
model_params:
  algorithm: "XGBoost"
  hyperparams:
    n_estimators: 100
    max_depth: 5
    learning_rate: 0.1
metrics: ["roc_auc", "pr_auc"]
output_dir: "models/"


Our train.py can parse this config (using PyYAML to load it) and feed those values into the pipeline. By abstracting configuration, we enable easy experimentation (e.g., switching model type or adjusting SMOTE settings without code changes). We also allow override via CLI flags for convenience.

CLI Entrypoints: The scripts/train.py and scripts/predict.py act as user-facing commands. For example, one can run python scripts/train.py --config configs/config.yaml from the command line to execute the whole pipeline. Internally these scripts will call the appropriate functions in src/. This design simulates a production CLI tool and makes it easy to integrate with job schedulers or simply for a user to run training. The predict.py could allow making predictions on a CSV of new prospects (for batch scoring) by loading the saved model – this complements the real-time API.

Logging: We incorporate Python’s logging to record info during pipeline execution (e.g., logging start/end of major steps, data shapes, metrics). In production, these logs would help monitor the pipeline runs and debug if something goes wrong. The log settings (level, format) can be configured via a logging.conf file in configs/.

In summary, the codebase is structured to be clean, extensible, and maintainable, following standard Python project practices. A new engineer could navigate the ml-sys repository and quickly find where each piece is implemented (data vs model vs API), and the presence of tests and type hints will further instill confidence.

Continuous Integration & Testing (CI/CD Pipeline)

To ensure production-level code quality, we set up Continuous Integration with GitHub Actions. The CI workflow (YAML configuration in .github/workflows/ci.yml) will run on each push or pull request, performing linting, formatting checks, and running our test suite automatically.

Linting with Ruff: We use Ruff, a fast Python linter, to catch code issues (unused imports, undefined variables, stylistic problems, etc.) and even enforce some formatting. The CI pipeline will install ruff and run it on the codebase as a step. For example, in the CI YAML:

- name: Run Ruff (Lint)
  run: ruff . --exit-zero  # (exit-zero if we want non-blocking lint, or remove for strict)


Ruff can also fix or format certain issues; we might configure it in pyproject.toml for consistency. (We could also use Black for formatting; note that Ruff has basic formatting capabilities too
docs.github.com
.)

Import Sorting with isort: Another job step runs isort to ensure imports are sorted (according to PEP8 grouping and alphabetical order). This keeps the code style tidy. We run isort --check . in CI to flag any files with unsorted imports (developers should run isort . locally to fix). This step will prevent merges if import orders are not compliant (ensuring consistency across contributions).

Unit Tests with Pytest: The CI then runs our test suite using pytest. We aim for robust coverage of critical components:

Data loading functions (do they correctly merge data? handle missing values?).

Preprocessing (does SMOTENC actually increase minority count? Are encoders working as expected?).

Model training (we might use a small subset of data and check that a model file is produced and metrics are reasonable or at least the training function executes without error).

API (using FastAPI’s TestClient to simulate requests: does /predict return a valid response for a sample input?).

Each push will trigger these tests, catching regressions or integration issues early. For instance, if a teammate breaks the interface between preprocessing and training, a unit test will fail. This fosters confidence that the codebase is reliable for production use.

Sample CI Workflow Snippet: A simplified example of the GitHub Actions workflow is:

name: CI
on: [push, pull_request]
jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v1
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: uv sync --dev
      - name: Lint with Ruff
        run: uv run ruff .          # Lint the entire codebase:contentReference[oaicite:34]{index=34}
      - name: Check import order
        run: uv run isort --check .
      - name: Run Unit Tests
        run: uv run pytest -q


Each step must succeed for the code to be considered mergeable. We also include badges in the README (e.g., build status) to show the CI pipeline status, signaling to the team that quality checks are in place.

Continuous Delivery (CD): While full deployment automation is beyond the scope here, we can extend the pipeline to CD in the future. For example, upon a new release/tag, we could have Actions build a Docker image of the FastAPI app and push to a registry, or deploy to a cloud service. The current CI ensures that at least our code is production-ready (linted, tested) for any manual deployment or further automation.

Finally, we also configure pre-commit hooks (optional) for developers, so that ruff/isort run locally before commits, aligning with the CI checks. This way, by the time code gets to GitHub, it's already formatted and linted correctly, smoothing the workflow.

Model Versioning Strategy

In a production ML system, it's crucial to version control the models just as we do code. Our design includes a simple but effective model versioning approach:

Semantic Versioning for Models: We tag each model with a version number like v1.0.0. The version can reflect changes in data or model (major = significant change or new data schema, minor = hyperparameter or architecture tweak, patch = minor bugfix). For example, our first model is conversion_model_v1.0.0.pkl. This version is referenced in the FastAPI service and in documentation. Adopting semantic versioning makes it clear which model is currently in production and how updates are incrementing
reddit.com
.

Artifact Storage: All model artifacts are stored in a dedicated location (models/ directory or an artifact store). We treat these as immutable artifacts – once a version is created and validated, it is never overwritten. This way, we can always roll back to a previous model if needed. In a larger team, one might use a model registry (like MLflow's registry or AWS SageMaker model registry), but for this project, a local storage with proper naming is sufficient.

Model Registry (Lightweight): We maintain a simple registry file (could be a YAML/JSON or even a Google Sheet in practice) that records each model version, metadata and status. For example, models/registry.json could list:

{
  "1.0.0": {"date": "2025-11-06", "roc_auc": 0.87, "pr_auc": 0.42, "features": ["ALEXA_RANK","INDUSTRY",...]},
  "1.1.0": {"date": "2026-01-10", "changes": "Added new feature X, retrained on Q4 data", "roc_auc": 0.89, ...}
}


Only one version is the production active version at a time (e.g., 1.0.0 initially). This registry helps in model governance, tracking when and how models were promoted. It also provides context (metrics, what changed).

Integration with Code: The FastAPI service can read from a config or environment variable which model version to load. For instance, an env var MODEL_VERSION=1.0.0 could be set when deploying. The load_model_artifacts() will then construct the filename using that version. This decouples code from a specific model file and makes upgrades safe and explicit. In CI/CD, when we are ready to promote a new model (say v1.1.0), we could update this env var or config and redeploy the service.

Promotion & Rollback: Only after a model passes evaluation criteria (e.g., no worse than the current model on key metrics, or vetted by stakeholders) do we “promote” it to production by updating the served version. Previous versions remain available. If an issue is found with a new model, we can quickly roll back by pointing the service back to the older version artifact (since it’s still stored and the code can load it). This approach aligns with best practices of treating models as data artifacts with semantic tags
reddit.com
, and avoids the pitfalls of ad-hoc file overwrites or unclear file names.

Experiment Tracking: While not fully implemented in this 5-hour scope, we note that using experiment tracking tools (MLflow, Weights & Biases, etc.) can further enhance versioning and reproducibility by logging parameters and data versions for each model run. In absence of that, we rely on our logs and registry to manually track what data/config was used for each model version. Each code git commit tied to a model training can be annotated (e.g., via commit message or git tag like model-v1.0.0) to link code and model versions.

By implementing model versioning, the ML lifecycle is managed more safely. The team can confidently iterate on models, knowing that any version can be referenced or deployed as needed and that the provenance of each model is documented.

Documentation and Usage Guidelines

Clear documentation is provided to make the project easy to understand and use. The main documentation is in the README.md, with the following key sections:

Project Scope & Background: Describes the problem (predicting prospect conversion for marketing prioritization) and the solution approach. We explain the business context in brief (HubSpot freemium model: prospects vs customers, conversion definition) and how this project helps (e.g., sales can focus on high-conversion probability prospects). This section also outlines the ML approach (classification model with usage data) in layman terms.

Data Description: Summarizes the dataset and features (drawing from the data dictionary provided). This helps readers unfamiliar with the data to grasp what inputs the model uses. We clarify any assumptions (e.g., usage data includes both pre- and post-conversion behavior for customers and how we handle that in features to avoid target leakage).

How to Install & Run: Step-by-step instructions to get the code running:

Installation: e.g., "Clone this repo, create a Python 3.10 virtual environment, and run uv sync --dev." If using Docker for reproducibility, we could provide a Dockerfile to build an image containing the environment.

Training the Model: e.g., "To train the model on the provided data, run python scripts/train.py --config configs/config.yaml. This will output a trained model file in models/ and print evaluation metrics." We mention any prerequisites (like ensure data CSVs are in a certain folder if not included in repo).

Evaluating Performance: If not already part of training output, explain how to run evaluation or that the training script already outputs ROC-AUC and PR-AUC. We might include sample metric results: "On the sample dataset, the model achieves ROC-AUC = 0.87 and PR-AUC = 0.43 on the test set." This gives the panel a sense of model quality (even though accuracy isn't the focus, it's good to show we measured it).

Serving Predictions: Instructions for the API. For example, "Run uvicorn src.api.app:app --reload to start the local server, then open a browser at http://localhost:8000/docs to view and test the API documentation. You can send a POST request to /predict with a JSON body containing company features. Example:

{
  \"industry\": \"Software\", 
  \"employee_range\": \"11-50\", 
  \"alexa_rank\": 500000, 
  \"actions_crm_contacts\": 10,
  ... 
}


The response will be a JSON with the conversion probability."

We also instruct how to use scripts/predict.py if one-off predictions on a file are supported (e.g., scoring a batch of prospects from a CSV and outputting a results CSV).

Design and Technical Decisions: This is a crucial section for the hiring panel. We explicitly list the key design decisions and rationales:

SMOTENC for class balancing: We explain that the positive class (conversion) was rare, so we used SMOTENC to synthetically augment it, which is preferable to naive oversampling or ignoring the imbalance. By using SMOTENC, we leveraged both numeric and categorical features in generating new examples
medium.com
, improving the model’s ability to learn minority class patterns. We caution how we avoided any information leakage by applying SMOTE only on training data
medium.com
.

Avoiding Train/Test Leakage: Emphasize measures like splitting data by time or ID before processing if needed, not using future data for past predictions, etc. For instance, since usage data included records before conversion for customers, one must be careful in feature engineering to not use post-conversion behavior to predict conversion. In our simple approach, we might restrict usage features to a period before conversion or include a flag if needed. This is mentioned to show awareness of potential leakage pitfalls.

Choice of XGBoost: We justify it as a proven algorithm for this kind of classification, offering a good balance of performance and interpretability (feature importance can be extracted). We also note we considered alternatives (Random Forest, Logistic Regression) but chose XGBoost for its boosting advantages and scalability. If hyperparameter tuning was minimal due to time, we mention what we would do with more time (e.g., use Optuna for HPO as demonstrated in similar churn projects
python.plainenglish.io
).

Pure Python Pipeline: We clarify why we avoided Airflow/Prefect: for a single-job pipeline that can be invoked via CLI, plain Python script is simpler and more transparent. However, we note the design is such that it could be integrated into an Airflow DAG or similar if the pipeline grew more complex (modular functions can be called in orchestrator tasks).

Config-driven and Modular Design: Highlight how using configs and modular code makes the framework flexible. A data scientist can add a new feature by editing the preprocessing module or config, without rewriting the whole pipeline. The modular design also makes testing easier (we can test each part in isolation).

CI/CD and Quality: We underline how automated linting (Ruff) and tests in GitHub Actions ensure code quality. This reflects a production mindset – any code that doesn't meet standards or breaks functionality won't be merged. We mention that we included unit tests even though the assessment might not require them, to show our commitment to reliable code.

Model Versioning: We describe how versioning is handled (as per the above section) – this shows foresight in how the model would be managed over time, which is a key production concern. We might even provide a short example in README of how one would promote a new model (e.g., updating an environment variable).

Future Improvements: Acknowledge what could be done with more time: e.g., more extensive feature engineering (incorporate time-based features from usage data), using a proper model registry or experiment tracker, implementing monitoring for the API (logging predictions, tracking data drift), or adding a batch inference pipeline for periodic re-scoring of all prospects. A brief discussion demonstrates we know how to take this further in a real-world setting even if not fully implemented now (the assessment expects discussion of what we'd do with more time, as a strengths-finding exercise).

How to Contribute/Extend: For completeness, we can include a note that developers can extend this project by following certain patterns. For instance, "to add a new data source or feature, add a function in src/data/ and include it in the data loading pipeline," or "to try a different model, implement a new trainer class in src/models/ and adjust the config." This aligns with making the framework useful for the team in the long run.

The README will be written in a clear, user-friendly tone, possibly with examples or command snippets (as illustrated above). If appropriate, we may include small tables or diagrams (like a high-level pipeline diagram) to illustrate the workflow.

Additionally, in-code documentation (docstrings) and auto-generated docs (FastAPI docs for the API, for example) complement the README. We ensure the API is self-documented via OpenAPI schema thanks to Pydantic models, fulfilling the requirement to communicate the API well.

Overall, the documentation ensures that anyone reviewing the project (like the hiring panel) or using it internally can understand what was built, why it was built that way, and how to use it or extend it. It demonstrates a production-readiness mindset, where maintainability and clarity are as important as functionality.

References:

HubSpot Finance MLE Assessment PDF – problem context and requirements

Imbalanced-learn documentation – SMOTENC for oversampling categorical data
medium.com
medium.com

Neptune.ai blog – importance of ROC-AUC and PR-AUC for imbalanced classification
neptune.ai
neptune.ai

FastAPI Churn Model example – FastAPI app structure for model serving
python.plainenglish.io
python.plainenglish.io

GitHub Actions Docs – using Ruff in CI for linting
docs.github.com

Reddit MLOps discussion – model versioning best practices (semantic versioning, artifact registry)

