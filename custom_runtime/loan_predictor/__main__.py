import argparse
import kserve

from .loan_predictor import LoanPredictor

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument(
    "--model_dir",
    help="Path to the AutoGluon TabularPredictor directory",
    required=True,
)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = LoanPredictor(name=args.model_name, model_dir=args.model_dir)
    model.load()
    kserve.ModelServer().start(models=[model])
