class ModelPredictor:
    def __init__(self):
        self.cnn_authority = 0.25
        self.mobilenet_authority = 0.75

    def predict(self, model_res):
        results = []

        # Iterate over the predictions for each label
        label_probs = {}
        for model, predictions in model_res.items():
            for prediction in predictions:
                label = prediction["label"]
                prob = prediction["prob"]
                # Initialize the label_probs dictionary if the label is not in it
                if label not in label_probs:
                    label_probs[label] = {"cnn_prob": 0, "mobilenet_prob": 0}

                # Update the probability for the corresponding model
                if model == "cnn":
                    label_probs[label]["cnn_prob"] = prob
                elif model == "mobilenet":
                    label_probs[label]["mobilenet_prob"] = prob

        # Calculate the weighted average for each label
        for label, probs in label_probs.items():
            weighted_avg_prob = (
                self.cnn_authority * probs["cnn_prob"] + self.mobilenet_authority * probs["mobilenet_prob"]
            )
            results.append({"label": label, "prob": weighted_avg_prob})

        # Sort the results by probability in descending order
        results.sort(key=lambda x: x["prob"], reverse=True)

        return results

# Sample usage
if __name__ == "__main__":
    model_res = {
        "cnn": [
            {"label": "Kohada", "prob": 0.92},
            {"label": "kos", "prob": 0.02},
            {"label": "Kohada", "prob": 0.002}
        ],
        "mobilenet": [
            {"label": "Kohada", "prob": 0.92},
            {"label": "kos", "prob": 0.02},
            {"label": "Kohada", "prob": 0.002}
        ]
    }



    predictor = ModelPredictor()
    results = predictor.predict(model_res)

    # Print the results
    for result in results:
        print(f"Label: {result['label']}, Probability: {result['prob']:.2f}")
