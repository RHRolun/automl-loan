# AutoML Loan Approval

Automated model training and deployment pipeline for a loan approval classifier using AutoGluon on OpenShift.

## Prerequisites

- OpenShift cluster with:
  - **OpenShift Pipelines** operator installed
  - **RHOAI 3.3 or later** operator installed
    - **KServe** enabled in RHOAI
- `oc` CLI
- Helm 3

## Install

```bash
helm install automl ./helm -n <YOUR_NAMESPACE> --set schedule="0 19 * * *" # Train a new model every day at 7pm
```

## What gets deployed

| Resource | Description |
|---|---|
| **MinIO** | S3-compatible storage with a `data` bucket pre-loaded with training data |
| **Workbench** | RHOAI Jupyter notebook with this repo cloned in at startup |
| **Tekton Pipeline** | `loan-model-pipeline` — triggered daily at 7pm via CronJob |

### Pipeline steps

`fetch-data` → `train` → `push-model` → `scan` → `sign` → `deploy-model`

Each run trains an AutoGluon TabularPredictor, packages the best model as an OCI modelcar, and deploys it as a KServe `InferenceService` named `loan-<timestamp>`.

## After install

**Explore and experiment** — open the workbench in the RHOAI dashboard and run `scripts/train.ipynb` for local training and `scripts/test_request.ipynb` to send inference requests to a deployed model.

**Trigger a pipeline run manually:**
```bash
oc create -f run.yaml -n <YOUR_NAMESPACE>
```

**Change the schedule** (default: daily at 7pm):
```bash
helm upgrade automl ./helm -n automl --set schedule="0 6 * * 1"  # Mondays at 6am
```
