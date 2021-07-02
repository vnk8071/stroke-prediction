from src.StrokeDetector import StrokeDetector

MODEL_PATH = '<dir to your model file>'
PARAM_PATH = '<dir to your model file>'
x = inputfromuser()

detector = StrokeDetector(modeltype=MODEL_PATH)

out = detector.predict(x)