# AutoML Loan Approval

Automated model training and deployment pipeline for a loan approval classifier using AutoGluon on OpenShift.

## Prerequisites

- OpenShift cluster with:
  - **OpenShift Pipelines** operator installed
  - **RHOAI 3.3 or later** operator installed
    - **KServe** enabled in RHOAI
- `oc` CLI, logged in with **cluster-admin**
- Helm 3

Enable the internal image registry's external route (required for pushing out modelcar):
```bash
oc patch configs.imageregistry.operator.openshift.io/cluster \
  --patch '{"spec":{"defaultRoute":true}}' --type=merge
```

## Install

```bash
helm install automl ./helm -n automl --create-namespace \
  --set dataUrl=https://raw.githubusercontent.com/RHRolun/automl-loan/main/cleaned_loan_data.csv
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

**Explore and experiment** — open the workbench in the RHOAI dashboard and run `train.ipynb` for local training and `test_request.ipynb` to send inference requests to a deployed model.

**Trigger a pipeline run manually:**
```bash
oc create -f tekton/run.yaml -n automl
```

**Change the schedule** (default: daily at 7pm):
```bash
helm upgrade automl ./helm -n automl --set schedule="0 6 * * 1"  # Mondays at 6am
```
