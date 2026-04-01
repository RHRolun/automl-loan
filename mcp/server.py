import os
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

mcp = FastMCP(
    "Loan Approval Predictor",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

INFER_ENDPOINT = os.environ["INFER_ENDPOINT"]   # e.g. https://loan-03191220-automl.apps.cluster.example.com
MODEL_NAME     = os.environ["MODEL_NAME"]        # e.g. loan-03191220


@mcp.tool()
def check_loan_approval(
    no_of_dependents: int,
    graduated: bool,
    self_employed: bool,
    income_annum: float,
    loan_amount: float,
    loan_term: int,
    cibil_score: int,
    residential_assets_value: float,
    commercial_assets_value: float,
    luxury_assets_value: float,
    bank_asset_value: float,
) -> str:
    """
    Check whether a loan application would be approved.

    Args:
        no_of_dependents: Number of dependents of the applicant
        graduated: Whether the applicant is a graduate
        self_employed: Whether the applicant is self-employed
        income_annum: Annual income of the applicant
        loan_amount: Requested loan amount
        loan_term: Loan term in months
        cibil_score: Credit score (300-900)
        residential_assets_value: Value of residential assets
        commercial_assets_value: Value of commercial assets
        luxury_assets_value: Value of luxury assets
        bank_asset_value: Value of bank assets
    """
    sample = {
        "no_of_dependents":         no_of_dependents,
        "graduated":                graduated,
        "self_employed":            self_employed,
        "income_annum":             income_annum,
        "loan_amount":              loan_amount,
        "loan_term":                loan_term,
        "cibil_score":              cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value":  commercial_assets_value,
        "luxury_assets_value":      luxury_assets_value,
        "bank_asset_value":         bank_asset_value,
    }

    payload = {
        "inputs": [
            {"name": col, "shape": [1], "datatype": "BYTES", "data": [val]}
            for col, val in sample.items()
        ]
    }

    url = f"{INFER_ENDPOINT}/v2/models/{MODEL_NAME}/infer"
    response = httpx.post(url, json=payload)
    response.raise_for_status()

    prediction = response.json()["outputs"][0]["data"][0]
    return "Loan approved" if bool(prediction) else "Loan rejected"


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
    mcp.run(transport=transport)
