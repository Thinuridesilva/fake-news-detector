import sys
sys.path.insert(0, 'src')

from api import health, predict, PredictRequest

print("Health:", health())

result = predict(PredictRequest(
    text="The Federal Reserve announced it would hold interest rates steady citing inflation data."
))
print("Baseline:", result.baseline)
print("DistilBERT:", result.distilbert)
print("Agreement:", result.agreement)
